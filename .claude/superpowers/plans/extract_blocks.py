"""Put a plan's code blocks in place, byte for byte.

    python3 extract_blocks.py PLAN PATH [PATH ...]

For each PATH the plan introduces with a line "Write `PATH`:" the block that
follows becomes the file (parents created, a final newline added). One
introduced with "Patch `PATH`:" is fed to `git apply` from the current
directory. A PATH the plan does not introduce exactly once is an error, and
nothing is written for any PATH after it.
"""

import re
import subprocess
import sys
from pathlib import Path

INTRO = re.compile(r"^(Write|Patch) `([^`]+)`:$")
FENCE = re.compile(r"^(`{3,})(\w*)$")


def blocks(plan):
    """{path: [(verb, text), ...]} for every introduced block, in plan order."""
    lines = Path(plan).read_text(encoding="utf-8").split("\n")
    found = {}
    i = 0
    while i < len(lines):
        intro = INTRO.match(lines[i].strip())
        if not intro:
            i += 1
            continue
        verb, path = intro.groups()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        fence = FENCE.match(lines[j]) if j < len(lines) else None
        if fence is None:
            sys.exit(f"{path}: no code block after its line {i + 1}")
        close = fence.group(1)
        k = j + 1
        while k < len(lines) and lines[k] != close:
            k += 1
        if k == len(lines):
            sys.exit(f"{path}: its code block never closes")
        found.setdefault(path, []).append((verb, "\n".join(lines[j + 1 : k]) + "\n"))
        i = k + 1
    return found


def main(plan, paths):
    found = blocks(plan)
    for path in paths:
        entries = found.get(path, [])
        if len(entries) != 1:
            sys.exit(f"{path} is introduced {len(entries)} times in {plan}, not once")
        verb, text = entries[0]
        if verb == "Write":
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print("wrote", path)
        else:
            subprocess.run(
                ["git", "apply", "--verbose", "-"], input=text.encode(), check=True
            )
            print("patched", path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
