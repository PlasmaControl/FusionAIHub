"""
Add data from source H5 files to a target H5 file (same shot).

Usage:
    python merge_h5.py target.h5 source1.h5 source2.h5 ...
    python merge_h5.py 200000.h5 200000_chiron.h5 200000_extra.h5
"""

import h5py
import sys
import argparse
from pathlib import Path


def add_to_h5(
        target_file: str | Path,
        source_files: list[str | Path],
        strategy: str = 'skip',
        verbose=True
):
    """
    Add trees/signals from source files to target file.

    Parameters
    ----------
    target_file: Path to target HDF5 file (modified in place)
        source_files: List of source HDF5 files to add from
        strategy: How to handle duplicates ('skip', 'overwrite', 'error')
        verbose: Print progress messages
    """
    if not Path(target_file).exists():
        print(f"Error: Target file does not exist: {target_file}")
        print("Create it first or use one of the source files as target")
        sys.exit(1)

    if verbose:
        print(f"Target file: {target_file}")

    with h5py.File(target_file, 'a') as f_target:
        stats = {
            'files_processed': 0,
            'trees_added': 0,
            'signals_added': 0,
            'signals_skipped': 0,
            'signals_overwritten': 0
        }

        for source_file in source_files:
            if not Path(source_file).exists():
                print(f"Warning: {source_file} does not exist, skipping")
                continue

            if Path(source_file).resolve() == Path(target_file).resolve():
                if verbose:
                    print(f"\nSkipping {source_file} (same as target)")
                continue

            if verbose:
                print(f"\nAdding from: {source_file}")

            try:
                with h5py.File(source_file, 'r') as f_source:
                    # Iterate over shots
                    for shot_name in f_source.keys():
                        if verbose:
                            print(f"  Shot {shot_name}:")

                        # Ensure shot exists in target
                        if shot_name not in f_target:
                            f_target.create_group(shot_name)
                            if verbose:
                                print(f"    Created shot group")

                        # Iterate over trees
                        for tree_name in f_source[shot_name].keys():
                            tree_path = f"{shot_name}/{tree_name}"

                            if tree_path not in f_target:
                                f_target.create_group(tree_path)
                                stats['trees_added'] += 1
                                if verbose:
                                    print(f"    Tree {tree_name} (new)")
                            else:
                                if verbose:
                                    print(f"    Tree {tree_name} (existing)")

                            # Iterate over signals
                            for signal_name in f_source[shot_name][tree_name].keys():
                                signal_path = f"{shot_name}/{tree_name}/{signal_name}"

                                # Check if signal exists
                                if signal_path in f_target:
                                    if strategy == 'skip':
                                        stats['signals_skipped'] += 1
                                        if verbose:
                                            print(f"      {signal_name} (skipped)")
                                        continue
                                    elif strategy == 'error':
                                        raise ValueError(
                                            f"Duplicate signal: {signal_path}")
                                    elif strategy == 'overwrite':
                                        del f_target[signal_path]
                                        stats['signals_overwritten'] += 1
                                        if verbose:
                                            print(f"      {signal_name} (overwritten)")

                                # Copy signal
                                f_source.copy(f_source[signal_path], f_target,
                                              signal_path)
                                stats['signals_added'] += 1

                                if verbose and strategy != 'skip':
                                    print(f"      {signal_name} (added)")

                stats['files_processed'] += 1

            except Exception as e:
                print(f"Error processing {source_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Trees added: {stats['trees_added']}")
        print(f"Signals added: {stats['signals_added']}")
        if stats['signals_skipped'] > 0:
            print(f"Signals skipped (duplicates): {stats['signals_skipped']}")
        if stats['signals_overwritten'] > 0:
            print(f"Signals overwritten: {stats['signals_overwritten']}")
        print(f"\nTarget file updated: {target_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Add data from source H5 files to target H5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """Examples:
        # Add chiron data to existing atlas file
        python merge_h5.py 200000_atlas.h5 200000_chiron.h5

        # Add multiple sources to target
        python merge_h5.py 200000.h5 200000_extra1.h5 200000_extra2.h5

        # Overwrite duplicates
        python merge_h5.py 200000.h5 200000_new.h5 --strategy overwrite

        Workflow:
        1. Fetch atlas data -> 200000_atlas.h5
        2. Fetch chiron data -> 200000_chiron.h5
        3. Merge: python merge_h5.py 200000_atlas.h5 200000_chiron.h5
        4. Result: 200000_atlas.h5 now contains both atlas and chiron trees
        """
    )

    parser.add_argument('target', help='Target HDF5 file (will be modified)')
    parser.add_argument('sources', nargs='+', help='Source HDF5 files to add')
    parser.add_argument(
        '--strategy', choices=['skip', 'overwrite', 'error'], default='skip',
        help='How to handle duplicate signals (default: skip)'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true', help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Add data
    add_to_h5(
        args.target,
        args.sources,
        strategy=args.strategy,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
