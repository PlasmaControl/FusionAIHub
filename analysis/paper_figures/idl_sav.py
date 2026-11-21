"""Write an IDL save file holding one anonymous structure of arrays.

scipy can read IDL ``.sav`` files but not write them. This writes the layout of the
colleague's camera files (``emission_structure_cam240paria_206879_raw.sav``, IDL 8.5.1,
uncompressed): a TIMESTAMP and a VERSION record, one VARIABLE record holding a scalar
anonymous structure whose tags are all arrays, and an END_MARKER. Arrays are given in
numpy (C) order and stored in IDL (column-major) order, so ``scipy.io.readsav`` returns
them with the same shape. Supported element types: uint8, int16, int32, float32, float64.

``write_struct(path, name, fields)`` with ``fields`` an ordered dict ``{tag: array}``.
``check_against(path)`` rewrites an existing file of this layout from its own contents
and reports the byte positions that differ (used to prove the format).
"""
import struct
import time
from pathlib import Path

import numpy as np

# IDL type code, element size, big-endian numpy dtype.
TYPES = {np.dtype(np.uint8): (1, 1, ">u1"), np.dtype(np.int16): (2, 2, ">i2"),
         np.dtype(np.int32): (3, 4, ">i4"), np.dtype(np.float32): (4, 4, ">f4"),
         np.dtype(np.float64): (5, 8, ">f8")}


def _long(v):
    return struct.pack(">l", int(v))


def _string(s):
    b = s.encode("latin1")
    return _long(len(b)) + b + b"\0" * (-len(b) % 4) if b else _long(0)


def _arraydesc(a, elsize):
    dims = list(a.shape[::-1]) + [1] * (8 - a.ndim)
    nbytes = a.size * elsize
    return (_long(8) + _long(elsize) + _long(nbytes) + _long(a.size) + _long(a.ndim)
            + _long(0) + _long(0) + _long(8) + b"".join(_long(d) for d in dims))


def _record(rectype, body, offset):
    nxt = offset + 16 + len(body)
    return struct.pack(">lIII", rectype, nxt % 2**32, nxt // 2**32, 0) + body


def write_struct(path, name, fields, timestamp=None, version=None):
    """Write ``fields`` as the tags of structure variable ``name``.

    ``timestamp`` and ``version`` are raw record bodies to copy (from a reference file);
    by default a fresh TIMESTAMP (date, user ``ignite``, host ``stellarai``) and IDL
    8.5.1's VERSION body are written.
    """
    arrays, descs, offs, off = [], [], [], 0
    for tag, a in fields.items():
        a = np.ascontiguousarray(a)
        code, elsize, be = TYPES[a.dtype.newbyteorder('=')]
        arrays.append((tag, a, code, elsize, be))
        offs.append(off)
        off += a.size * elsize
    total = off
    # Structure descriptor: STRUCTSTART 9, empty name (anonymous), flags 8, ntags,
    # nbytes 0, then per tag (offset, type, flags 0x14 = array), names, array descriptors.
    sd = _long(9) + _string("") + _long(8) + _long(len(arrays)) + _long(0)
    for (tag, a, code, elsize, be), o in zip(arrays, offs):
        sd += _long(o) + _long(code) + _long(0x14)
    for tag, *_ in arrays:
        sd += _string(tag)
    for tag, a, code, elsize, be in arrays:
        sd += _arraydesc(a, elsize)
    # The structure itself is a 1-element array of itself (element size = total bytes).
    head = (_string(name.upper()) + _long(8) + _long(0x34)
            + _long(8) + _long(total) + _long(total) + _long(1) + _long(1)
            + _long(0) + _long(0) + _long(8) + b"".join(_long(1) for _ in range(8)))
    if timestamp is None:
        timestamp = (b"\0" * 1024 + _string(time.strftime("%a %b %e %H:%M:%S %Y"))
                     + _string("ignite") + _string("stellarai"))
    if version is None:
        version = _long(12) + _string("x86_64") + _string("linux") + _string("8.5.1")
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(b"SR\x00\x04")
        pos = 4
        for rectype, body in ((10, timestamp), (14, version)):
            rec = _record(rectype, body, pos)
            f.write(rec)
            pos += len(rec)
        # VARIABLE record: header, descriptors, VARSTART 7, then each tag's data
        # (a byte array is preceded by its byte count), each padded to 4 bytes.
        body_len = len(head) + len(sd) + 4
        for tag, a, code, elsize, be in arrays:
            n = a.size * elsize
            body_len += (4 if code == 1 else 0) + n + (-n % 4)
        nxt = pos + 16 + body_len
        f.write(struct.pack(">lIII", 2, nxt % 2**32, nxt // 2**32, 0))
        f.write(head + sd + _long(7))
        for tag, a, code, elsize, be in arrays:
            # IDL is column-major: the C-order bytes of a are the IDL array with the
            # dimensions reversed, which is what the descriptor states.
            data = a.astype(be, copy=False).tobytes(order="C")
            if code == 1:
                f.write(_long(len(data)))
            f.write(data)
            f.write(b"\0" * (-len(data) % 4))
        assert f.tell() == nxt, (f.tell(), nxt)
        f.write(struct.pack(">lIII", 6, 0, 0, 0))
    tmp.replace(path)


def _records(b):
    """(rectype, start, end) of each record of an uncompressed save file."""
    out, pos = [], 4
    while True:
        rt, lo, hi, _ = struct.unpack(">lIII", b[pos:pos + 16])
        nxt = lo + hi * 2**32
        out.append((rt, pos, nxt))
        if rt == 6:
            return out
        pos = nxt


def check_against(ref, out):
    """Rewrite ``ref`` from its own readsav contents into ``out``; return differing bytes."""
    import scipy.io

    b = Path(ref).read_bytes()
    recs = {rt: (s, e) for rt, s, e in _records(b)}
    ts = b[recs[10][0] + 16:recs[10][1]]
    ver = b[recs[14][0] + 16:recs[14][1]]
    d = scipy.io.readsav(ref)
    (name, rec), = d.items()
    s = rec[0]
    fields = {t: np.asarray(s[t]) for t in rec.dtype.names}
    write_struct(out, name, fields, timestamp=ts, version=ver)
    c = Path(out).read_bytes()
    if len(b) != len(c):
        return f"length {len(b)} vs {len(c)}"
    diff = np.flatnonzero(np.frombuffer(b, np.uint8) != np.frombuffer(c, np.uint8))
    return diff
