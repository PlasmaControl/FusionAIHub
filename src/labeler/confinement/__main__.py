"""The confinement workflow: labels, export, data, train, apply, elm, comparison."""

import importlib
import sys

from . import run_directory


def main():
    commands = ("labels", "export", "data", "train", "apply", "elm", "comparison")
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(
            "Usage: python -m labeler.confinement {"
            + ",".join(commands)
            + "} [options]"
        )
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2
    module = importlib.import_module(f"labeler.confinement.{sys.argv[1]}")
    # Redirect outputs only; preserve shared read-only diagnostic inputs and the
    # frozen extraction/fitting code used for this experiment.
    flags = {
        "labels": ("--out",),
        "export": ("--run-dir",),
        "data": ("--labels", "--out"),
        "train": ("--features", "--out"),
        "apply": ("--model-dir",),
        "elm": ("--run-dir",),
        "comparison": ("--run-dir",),
    }
    argv = sys.argv[2:]
    for flag in flags[sys.argv[1]]:
        if not any(arg == flag or arg.startswith(flag + "=") for arg in argv):
            argv += [flag, str(run_directory())]
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
