"""Camera previews in the atomic review store, never whole movies in the page.

Corpus clocks are seconds; stored and served clocks are milliseconds. Only
image arrays (T,H,W) or (C,T,H,W) are movies: the usual bolo (48,T) is a
diagnostic trace, not a tomographic image. Each channel keeps its native clock,
decimated to at most 20 fps, and one fixed grayscale scale over the shot.
"""

from __future__ import annotations

import io
import json
import math

import h5py
import numpy as np
from PIL import Image

CAMERAS = {"bolo": (80, 120), "tangtv": (240, 720), "irtv": (256, 320)}
MAX_FPS = 20.0
PERCENTILES = (1.0, 99.5)
# Active input_key order in data/config/modalities/modalities.yaml. IRTV declares
# seven output slots but only SIX active nodes; PERI75R0 is commented out. Do not
# shift UPCEN/UPDIV to manufacture a seventh mapping.
VIEW_NODES = {
    "tangtv": [
        ("LODIV_240RM1:PAR:INTENSIFIED", "lower divertor"),
        ("LODIV_240RM1:PAR:STANDARD", "lower divertor"),
        ("LODIV_240RM1:PERP:STANDARD", "lower divertor"),
        ("UPDIV_225RP1:PERP:STANDARD", "upper divertor"),
        ("UPDIV_0RP1:PERP:STANDARD", "upper divertor"),
        ("UPDIV_225RP1:PAR:STANDARD", "upper divertor"),
        ("UPDIV_0RP1:PAR:STANDARD", "upper divertor"),
    ],
    "irtv": [
        ("BIAS_105RM1", "bias view"),
        ("LOCEN_315RM1", "lower central"),
        ("LODIV_165RP2", "lower divertor"),
        ("LODIV_60RP2", "lower divertor"),
        ("UPCEN_300RP1", "upper central"),
        ("UPDIV_225RM2", "upper divertor"),
        ("unmapped padded slot", "unknown"),
    ],
}


def view(camera, channel):
    """Physical identity, without guessing unrecorded spectral filters."""
    nodes = VIEW_NODES.get(camera, [])
    name, region = (
        nodes[channel] if channel < len(nodes) else (f"image {channel}", "unknown")
    )
    node = None
    if camera in VIEW_NODES and name != "unmapped padded slot":
        prefix = camera.upper()
        suffix = "VIDEO_IMAGES" if camera == "tangtv" else "DIGITAL_CAM:DIGITAL_RAW"
        node = f"\\{prefix}::TOP.{prefix}:{name}:{suffix}"
    return {"channel": channel, "view_name": name, "region": region, "node": node}


def frame_indices(times, max_fps=MAX_FPS) -> np.ndarray:
    """Finite, increasing native times separated by at least 1/max_fps seconds.

    Duplicate/backwards clocks are rejected rather than silently reordered.
    Nonfinite timestamps are dropped, with their corresponding frames.
    """
    if not math.isfinite(max_fps) or max_fps <= 0:
        raise ValueError("max_fps must be positive and finite")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("camera clock must be one-dimensional")
    finite = np.flatnonzero(np.isfinite(times))
    if len(finite) < 2:
        return np.array([], dtype=int)
    if np.any(np.diff(times[finite]) <= 0):
        raise ValueError("camera clock must be strictly increasing")
    kept = [int(finite[0])]
    for i in finite[1:]:
        if times[i] - times[kept[-1]] >= 1 / max_fps - 1e-9:
            kept.append(int(i))
    return np.asarray(kept, dtype=int)


def _layout(group):
    if "xdata" not in group or "ydata" not in group:
        raise ValueError("camera has no xdata/ydata")
    times = np.asarray(group["xdata"], dtype=float)
    data = group["ydata"]
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("one-sample camera stub")
    if data.ndim not in (3, 4):
        raise ValueError("group contains traces, not image frames")
    axis = 1 if data.ndim == 4 else 0
    if data.shape[axis] != len(times) or min(data.shape[-2:]) < 1:
        raise ValueError("camera shape and clock disagree")
    if min(data.shape[-2:]) == 1:
        raise ValueError("flattened camera array; image geometry is unavailable")
    return times, data, data.shape[0] if data.ndim == 4 else 1


def _frame(data, channel, index, shape):
    stride = max(
        max(1, math.ceil(n / limit))
        for n, limit in zip(data.shape[-2:], shape, strict=True)
    )
    key = (channel, index) if data.ndim == 4 else (index,)
    frame = np.asarray(data[key], dtype=np.float32)
    if stride == 1:
        return frame
    h, w = frame.shape
    # Reduce actual pixels without padding a narrow image to a huge square.
    # Partial edge blocks use their actual finite count; aspect uses one stride.
    starts_y, starts_x = np.arange(0, h, stride), np.arange(0, w, stride)
    finite = np.isfinite(frame)

    def area(values):
        return np.add.reduceat(
            np.add.reduceat(values, starts_y, axis=0), starts_x, axis=1
        )

    sums = area(np.where(finite, frame, 0))
    counts = area(finite.astype(np.int32))
    mean = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean


def write(store, corpus) -> None:
    """Stream previews into an open review HDF5 file, bounded to one frame.

    Percentiles use a deterministic pixel sample of every selected frame.
    All-NaN channels/frames are omitted, retaining original source indices.
    Missing cameras are recorded so the UI can explain their absence.
    """
    videos = store.create_group("videos")
    videos.attrs["max_fps"] = MAX_FPS
    videos.attrs["scale_percentiles"] = json.dumps(PERCENTILES)
    source = h5py.File(corpus, "r") if corpus.is_file() else None
    try:
        for camera, shape in CAMERAS.items():
            target = videos.create_group(camera)
            target.attrs["reason"] = "camera group missing"
            if source is None or camera not in source:
                continue
            try:
                times, data, channels = _layout(source[camera])
                indices = frame_indices(times)
            except ValueError as error:
                target.attrs["reason"] = str(error)
                continue
            target.attrs["source_shape"] = json.dumps(data.shape)
            target.attrs["source_frames"] = len(times)
            live = 0
            for channel in range(channels):
                samples, valid = [], []
                for index in indices:
                    frame = _frame(data, channel, int(index), shape)
                    finite = frame[np.isfinite(frame)]
                    if not finite.size:
                        continue
                    valid.append(int(index))
                    samples.append(finite[:: max(1, math.ceil(finite.size / 2048))])
                if not valid:
                    continue
                lo, hi = np.percentile(np.concatenate(samples), PERCENTILES)
                lo, hi = float(lo), float(hi)
                if hi <= lo:
                    hi = lo + 1.0
                frames = target.create_group(str(channel))
                frames.attrs.update({"z_lo": lo, "z_hi": hi})
                frames.create_dataset("times_ms", data=times[valid] * 1000)
                frames.create_dataset("source_indices", data=valid)
                h, w = _frame(data, channel, valid[0], shape).shape
                output = frames.create_dataset(
                    "frames",
                    (len(valid), h, w),
                    dtype="uint8",
                    chunks=(1, h, w),
                    compression="gzip",
                    compression_opts=1,
                )
                for j, index in enumerate(valid):
                    frame = _frame(data, channel, index, shape)
                    scaled = (
                        np.nan_to_num(frame, nan=lo, posinf=hi, neginf=lo).astype(
                            np.float64
                        )
                        - lo
                    ) * (255 / (hi - lo))
                    output[j] = np.clip(np.rint(scaled), 0, 255).astype("uint8")
                live += 1
            target.attrs["reason"] = "" if live else "no finite image frames"
    finally:
        if source is not None:
            source.close()


def meta(path) -> dict:
    """Small manifests and frame clocks only; pixel data stays on disk."""
    cameras = []
    with h5py.File(path, "r") as store:
        if "videos" not in store:
            return {"cameras": []}
        lo = float(store.attrs["t0_ms"])
        hi = lo + float(store.attrs["dt_ms"]) * int(store.attrs["n"])
        for camera in CAMERAS:
            group = store["videos"][camera]
            channels = []
            for key in sorted(group, key=int):
                frames = group[key]
                channels.append(
                    {
                        **view(camera, int(key)),
                        "times_ms": frames["times_ms"][:].tolist(),
                        "shape": list(frames["frames"].shape[1:]),
                        "scale": [float(frames.attrs[n]) for n in ("z_lo", "z_hi")],
                    }
                )
            preferred = {"tangtv": (0, 2), "irtv": (2, 3)}.get(camera, ())
            in_window = [
                ch for ch in channels if any(lo <= t <= hi for t in ch["times_ms"])
            ]
            default = next(
                (ch for c in preferred for ch in in_window if ch["channel"] == c), None
            )
            if default is None:
                default = next(
                    (ch for c in preferred for ch in channels if ch["channel"] == c),
                    None,
                )
            if default is None:
                default = next(iter(in_window or channels), None)
            cameras.append(
                {
                    "name": camera,
                    "views": [
                        view(camera, c) for c in range(len(VIEW_NODES.get(camera, [])))
                    ],
                    "channels": channels,
                    "reason": str(group.attrs["reason"]),
                    "default_channel": default["channel"] if default else None,
                    "spectral_note": "Filter/emission line is not recorded in the corpus.",
                    **(
                        {
                            "inactive_node": "\\IRTV::TOP.IRTV:PERI75R0:DIGITAL_CAM:DIGITAL_RAW",
                            "mapping_note": "Six active nodes; slot 6 has no configured node. PERI75R0 is commented out.",
                        }
                        if camera == "irtv"
                        else {}
                    ),
                }
            )
    return {"cameras": cameras, "max_fps": MAX_FPS}


def nearest_index(times, t_ms: float) -> int:
    """Nearest native frame, endpoints clamped; an exact tie chooses earlier."""
    if not math.isfinite(t_ms):
        raise ValueError("frame time must be finite")
    times = np.asarray(times, dtype=float)
    if not len(times):
        raise KeyError("no frames for this camera")
    right = min(int(np.searchsorted(times, t_ms)), len(times) - 1)
    if right and abs(t_ms - times[right - 1]) <= abs(times[right] - t_ms):
        right -= 1
    return right


def read_frame(path, camera: str, channel: int, t_ms: float) -> tuple[bytes, dict]:
    """Read and encode only the nearest preview as grayscale PNG."""
    if camera not in CAMERAS or channel < 0:
        raise KeyError("unknown camera/channel")
    with h5py.File(path, "r") as store:
        group = store[f"videos/{camera}/{channel}"]
        times = group["times_ms"][:]
        index = nearest_index(times, t_ms)
        frame = group["frames"][index]
    encoded = io.BytesIO()
    Image.fromarray(frame).save(encoded, format="PNG")
    return encoded.getvalue(), {"index": index, "time_ms": float(times[index])}
