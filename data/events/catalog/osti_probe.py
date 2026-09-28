"""Read-only probe: which DIII-D shot numbers appear in OSTI full text next to "DIII-D".

Usage: python3 osti_probe.py <shots.txt> <out.jsonl>
Resumable: shots already in out.jsonl are skipped. One request at a time, ~1/s.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.osti.gov/api/v1/records"
UA = "FusionAIHub-literature-probe/0.1 (research; low rate)"


def query(shot, rows=100, tries=5):
    params = {"fulltext": '"%d" AND "DIII-D"' % shot, "rows": str(rows)}
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 5.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("x-total-count") or 0)
                body = json.loads(r.read().decode("utf-8") or "[]")
            return total, body
        except Exception as exc:  # noqa: BLE001 - probe script, keep going
            if attempt == tries - 1:
                return None, str(exc)
            time.sleep(delay)
            delay *= 2
    return None, "unreachable"


def slim(rec):
    return {
        "osti_id": rec.get("osti_id"),
        "doi": rec.get("doi"),
        "title": rec.get("title"),
        "date": (rec.get("publication_date") or "")[:10],
        "journal": rec.get("journal_name"),
        "type": rec.get("product_type"),
    }


def main():
    shots_path, out_path = sys.argv[1], sys.argv[2]
    shots = [int(x) for x in open(shots_path).read().split()]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["shot"])
            except Exception:  # noqa: BLE001
                pass
    todo = [s for s in shots if s not in done]
    print("shots %d, done %d, todo %d" % (len(shots), len(done), len(todo)), flush=True)
    with open(out_path, "a") as out:
        for i, shot in enumerate(todo):
            total, body = query(shot)
            if total is None:
                row = {"shot": shot, "error": body}
            else:
                row = {"shot": shot, "n": total, "records": [slim(r) for r in body]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 250 == 0:
                print("%d/%d shot %d n=%s" % (i, len(todo), shot, row.get("n")), flush=True)
            time.sleep(0.8)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
