#!/usr/bin/env python3
"""
Add/overwrite data from source H5 files to target H5 file.
Existing signals are overwritten, new signals are added.

Usage:
    python add_to_h5.py target.h5 source1.h5 source2.h5 ...
    python add_to_h5.py 200000.h5 200000_chiron.h5 200000_extra.h5
"""

import h5py
import sys
import argparse
from pathlib import Path


def add_to_h5(target_file, source_files, verbose=True):
    """
    Add/overwrite trees/signals from source files to target file.

    Args:
        target_file: Path to target HDF5 file (modified in place)
        source_files: List of source HDF5 files to add from
        verbose: Print progress messages
    """
    if not Path(target_file).exists():
        print(f"Error: Target file does not exist: {target_file}")
        print("Create it first or use one of the source files as target")
        sys.exit(1)

    if verbose:
        print(f"Target file: {target_file}")
        print(f"Mode: Overwrite existing signals, add new ones\n")

    with h5py.File(target_file, 'a') as f_target:
        stats = {
            'files_processed': 0,
            'trees_added': 0,
            'signals_added': 0,
            'signals_overwritten': 0
        }

        for source_file in source_files:
            if not Path(source_file).exists():
                print(f"Warning: {source_file} does not exist, skipping")
                continue

            if Path(source_file).resolve() == Path(target_file).resolve():
                if verbose:
                    print(f"Skipping {source_file} (same as target)")
                continue

            if verbose:
                print(f"Adding from: {source_file}")

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
                                    # Overwrite
                                    del f_target[signal_path]
                                    f_source.copy(f_source[signal_path], f_target,
                                                  signal_path)
                                    stats['signals_overwritten'] += 1
                                    if verbose:
                                        print(f"      {signal_name} (overwritten)")
                                else:
                                    # Add new
                                    f_source.copy(f_source[signal_path], f_target,
                                                  signal_path)
                                    stats['signals_added'] += 1
                                    if verbose:
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
        print(f"Signals overwritten: {stats['signals_overwritten']}")
        print(f"\nTarget file updated: {target_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Add/overwrite data from source H5 files to target H5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add/update chiron data to existing file
  python add_to_h5.py 200000.h5 200000_chiron.h5

  # Add multiple sources (later sources overwrite earlier ones)
  python add_to_h5.py 200000.h5 source1.h5 source2.h5 source3.h5

  # Update all files in directory with new data
  for file in *.h5; do 
      python add_to_h5.py "$file" updated_data.h5
  done

Behavior:
  - If signal exists in target: OVERWRITE with source data
  - If signal is new: ADD to target
  - Trees are merged (not replaced entirely)
        """
    )

    parser.add_argument('target', help='Target HDF5 file (will be modified)')
    parser.add_argument('sources', nargs='+',
                        help='Source HDF5 files to add/overwrite from')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    # Add/overwrite data
    add_to_h5(
        args.target,
        args.sources,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
