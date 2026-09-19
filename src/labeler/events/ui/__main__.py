"""Run the browser verification surface."""

import argparse

from .serve import DEFAULT_PORT, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="labeler.events.ui")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    raise SystemExit(main(port=args.port, token=args.token))
