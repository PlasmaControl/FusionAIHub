#!/usr/bin/env python3
"""
Rename groups in HDF5 files.
If new name exists, checks if data is identical before proceeding.

Usage:
    python rename_h5_groups.py file.h5 old_name new_name [--level shot|tree|signal]
    python rename_h5_groups.py 200000.h5 d3d D3D --level tree
    python rename_h5_groups.py 200000.h5 200000 200001 --level shot
"""

import h5py
import sys
import argparse
import numpy as np
from pathlib import Path


def compare_groups(group1, group2, path=""):
    """
    Recursively compare two HDF5 groups for equality.

    Args:
        group1: First HDF5 group
        group2: Second HDF5 group
        path: Current path (for error messages)

    Returns:
        (is_equal, differences) tuple
    """
    differences = []

    # Check if both have same keys
    keys1 = set(group1.keys())
    keys2 = set(group2.keys())

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            differences.append(f"{path}: Only in first: {only_in_1}")
        if only_in_2:
            differences.append(f"{path}: Only in second: {only_in_2}")
        return False, differences

    # Compare each key
    for key in keys1:
        item1 = group1[key]
        item2 = group2[key]
        current_path = f"{path}/{key}" if path else key

        # Check if both are same type (group vs dataset)
        if isinstance(item1, h5py.Group) != isinstance(item2, h5py.Group):
            differences.append(f"{current_path}: Type mismatch")
            return False, differences

        if isinstance(item1, h5py.Group):
            # Recursively compare subgroups
            equal, subdiffs = compare_groups(item1, item2, current_path)
            if not equal:
                differences.extend(subdiffs)
                return False, differences
        else:
            # Compare datasets
            try:
                # Handle scalar vs array datasets
                if item1.shape == ():
                    # Scalar dataset - use [()] instead of [:]
                    data1 = item1[()]
                    data2 = item2[()]
                else:
                    # Array dataset
                    data1 = item1[:]
                    data2 = item2[:]

                # Check shapes
                if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
                    if data1.shape != data2.shape:
                        differences.append(
                            f"{current_path}: Shape mismatch {data1.shape} vs {data2.shape}")
                        return False, differences

                # Check data equality (handle NaNs)
                if isinstance(data1, (np.ndarray, np.floating, float)):
                    # For float data, use allclose and handle NaNs
                    if isinstance(data1, np.ndarray):
                        if np.issubdtype(data1.dtype, np.floating):
                            nan_mask1 = np.isnan(data1)
                            nan_mask2 = np.isnan(data2)

                            if not np.array_equal(nan_mask1, nan_mask2):
                                differences.append(
                                    f"{current_path}: NaN positions differ")
                                return False, differences

                            # Compare non-NaN values
                            if np.any(nan_mask1):
                                # Has NaNs
                                if not np.allclose(data1[~nan_mask1], data2[~nan_mask2],
                                                   rtol=1e-9, atol=1e-9):
                                    differences.append(
                                        f"{current_path}: Data values differ")
                                    return False, differences
                            else:
                                # No NaNs
                                if not np.allclose(data1, data2, rtol=1e-9, atol=1e-9):
                                    differences.append(
                                        f"{current_path}: Data values differ")
                                    return False, differences
                        else:
                            # Non-float array
                            if not np.array_equal(data1, data2):
                                differences.append(f"{current_path}: Data values differ")
                                return False, differences
                    else:
                        # Scalar float/number
                        if np.isnan(data1) and np.isnan(data2):
                            pass  # Both NaN, equal
                        elif np.isnan(data1) or np.isnan(data2):
                            differences.append(
                                f"{current_path}: One is NaN, other is not")
                            return False, differences
                        elif not np.isclose(data1, data2, rtol=1e-9, atol=1e-9):
                            differences.append(
                                f"{current_path}: Data values differ ({data1} vs {data2})")
                            return False, differences
                elif isinstance(data1, bytes) and isinstance(data2, bytes):
                    # String/bytes comparison
                    if data1 != data2:
                        differences.append(f"{current_path}: String values differ")
                        return False, differences
                else:
                    # General comparison
                    if data1 != data2:
                        differences.append(f"{current_path}: Data values differ")
                        return False, differences

                # Check attributes
                attrs1 = dict(item1.attrs)
                attrs2 = dict(item2.attrs)

                # Compare attributes (handle different types)
                if set(attrs1.keys()) != set(attrs2.keys()):
                    differences.append(f"{current_path}: Attribute keys differ")
                    return False, differences

                for attr_key in attrs1.keys():
                    val1 = attrs1[attr_key]
                    val2 = attrs2[attr_key]

                    # Convert to comparable types
                    if isinstance(val1, bytes):
                        val1 = val1.decode('utf-8') if isinstance(val1, bytes) else val1
                    if isinstance(val2, bytes):
                        val2 = val2.decode('utf-8') if isinstance(val2, bytes) else val2

                    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                        if not np.array_equal(val1, val2):
                            differences.append(
                                f"{current_path}: Attribute '{attr_key}' differs")
                            return False, differences
                    else:
                        if val1 != val2:
                            differences.append(
                                f"{current_path}: Attribute '{attr_key}' differs ({val1} vs {val2})")
                            return False, differences

            except Exception as e:
                differences.append(f"{current_path}: Comparison error: {e}")
                return False, differences

    # Check group-level attributes
    attrs1 = dict(group1.attrs)
    attrs2 = dict(group2.attrs)

    if set(attrs1.keys()) != set(attrs2.keys()):
        differences.append(f"{path}: Group attribute keys differ")
        return False, differences

    for attr_key in attrs1.keys():
        val1 = attrs1[attr_key]
        val2 = attrs2[attr_key]

        if isinstance(val1, bytes):
            val1 = val1.decode('utf-8') if isinstance(val1, bytes) else val1
        if isinstance(val2, bytes):
            val2 = val2.decode('utf-8') if isinstance(val2, bytes) else val2

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                differences.append(f"{path}: Group attribute '{attr_key}' differs")
                return False, differences
        else:
            if val1 != val2:
                differences.append(f"{path}: Group attribute '{attr_key}' differs")
                return False, differences

    return True, []


def rename_group(h5_file, old_name, new_name, level='tree', shot=None, tree=None,
                 dry_run=False, verbose=True):
    """
    Rename a group in HDF5 file.

    If new_name already exists, compares data. If identical, removes old_name.
    If different, raises error.

    Args:
        h5_file: Path to HDF5 file
        old_name: Current name of group
        new_name: New name for group
        level: 'shot', 'tree', or 'signal'
        shot: Shot number (required for tree/signal level)
        tree: Tree name (required for signal level)
        dry_run: Show what would be renamed without doing it
        verbose: Print progress
    """
    if old_name == new_name:
        if verbose:
            print(f"Warning: old_name and new_name are identical ('{old_name}')")
            print("Nothing to do.")
        return 0

    if not Path(h5_file).exists():
        print(f"Error: File does not exist: {h5_file}")
        sys.exit(1)

    mode = 'r' if dry_run else 'a'

    with h5py.File(h5_file, mode) as f:
        renamed = 0

        if level == 'shot':
            # Rename shot: 200000 -> 200001
            if old_name not in f:
                print(f"Error: Shot '{old_name}' not found")
                return 0

            if new_name in f:
                # Compare data
                if verbose:
                    print(f"Shot '{new_name}' already exists, comparing data...")

                is_equal, differences = compare_groups(f[old_name], f[new_name],
                                                       old_name)

                if is_equal:
                    if verbose:
                        print(f"  ✓ Data is identical")
                        print(f"  Removing duplicate shot: {old_name}")

                    if not dry_run:
                        del f[old_name]

                    renamed = 1
                else:
                    print(f"  ✗ Data is different:")
                    for diff in differences[:5]:  # Show first 5 differences
                        print(f"    - {diff}")
                    if len(differences) > 5:
                        print(f"    ... and {len(differences) - 5} more differences")
                    print(f"Error: Cannot rename - data conflict")
                    return 0
            else:
                # Normal rename
                if verbose:
                    print(f"Renaming shot: {old_name} -> {new_name}")

                if not dry_run:
                    f.copy(f[old_name], f, new_name)
                    del f[old_name]

                renamed = 1

        elif level == 'tree':
            # Rename tree in all shots or specific shot
            shots_to_process = [shot] if shot else list(f.keys())

            for shot_name in shots_to_process:
                if shot_name not in f:
                    print(f"Warning: Shot '{shot_name}' not found")
                    continue

                if old_name not in f[shot_name]:
                    if verbose:
                        print(f"Shot {shot_name}: Tree '{old_name}' not found")
                    continue

                old_path = f"{shot_name}/{old_name}"
                new_path = f"{shot_name}/{new_name}"

                if new_name in f[shot_name]:
                    # Compare data
                    if verbose:
                        print(
                            f"Shot {shot_name}: Tree '{new_name}' already exists, comparing data...")

                    is_equal, differences = compare_groups(f[old_path], f[new_path],
                                                           old_path)

                    if is_equal:
                        if verbose:
                            print(f"  ✓ Data is identical")
                            print(f"  Removing duplicate tree: {old_name}")

                        if not dry_run:
                            del f[old_path]

                        renamed += 1
                    else:
                        print(f"  ✗ Data is different:")
                        for diff in differences[:5]:
                            print(f"    - {diff}")
                        if len(differences) > 5:
                            print(f"    ... and {len(differences) - 5} more differences")
                        print(
                            f"Error: Cannot rename tree in shot {shot_name} - data conflict")
                else:
                    # Normal rename
                    if verbose:
                        print(
                            f"Shot {shot_name}: Renaming tree {old_name} -> {new_name}")

                    if not dry_run:
                        f.copy(f[old_path], f[shot_name], new_name)
                        del f[old_path]

                    renamed += 1

        elif level == 'signal':
            # Rename signal in specific tree
            if not shot or not tree:
                print("Error: --shot and --tree required for signal level")
                return 0

            tree_path = f"{shot}/{tree}"

            if tree_path not in f:
                print(f"Error: Tree '{tree_path}' not found")
                return 0

            if old_name not in f[tree_path]:
                print(f"Error: Signal '{old_name}' not found in {tree_path}")
                return 0

            old_path = f"{tree_path}/{old_name}"
            new_path = f"{tree_path}/{new_name}"

            if new_name in f[tree_path]:
                # Compare data
                if verbose:
                    print(f"Signal '{new_name}' already exists, comparing data...")

                is_equal, differences = compare_groups(f[old_path], f[new_path],
                                                       old_path)

                if is_equal:
                    if verbose:
                        print(f"  ✓ Data is identical")
                        print(f"  Removing duplicate signal: {old_name}")

                    if not dry_run:
                        del f[old_path]

                    renamed = 1
                else:
                    print(f"  ✗ Data is different:")
                    for diff in differences[:5]:
                        print(f"    - {diff}")
                    if len(differences) > 5:
                        print(f"    ... and {len(differences) - 5} more differences")
                    print(f"Error: Cannot rename signal - data conflict")
                    return 0
            else:
                # Normal rename
                if verbose:
                    print(f"Renaming signal: {old_path} -> {new_path}")

                if not dry_run:
                    f.copy(f[old_path], f[tree_path], new_name)
                    del f[old_path]

                renamed = 1

        if verbose and renamed > 0:
            if dry_run:
                print(f"\nDry run: Would rename/remove {renamed} group(s)")
            else:
                print(f"\nRenamed/removed {renamed} group(s) in {h5_file}")

        return renamed


def batch_rename(h5_file, mapping_file, level='tree', dry_run=False, verbose=True):
    """
    Rename multiple groups from a mapping file.

    Args:
        h5_file: Path to HDF5 file
        mapping_file: Path to text file with "old_name new_name" pairs
        level: 'shot', 'tree', or 'signal'
        dry_run: Show what would be renamed
        verbose: Print progress
    """
    if not Path(mapping_file).exists():
        print(f"Error: Mapping file does not exist: {mapping_file}")
        sys.exit(1)

    total_renamed = 0

    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Invalid line (expected 'old new'): {line}")
                continue

            old_name, new_name = parts
            renamed = rename_group(h5_file, old_name, new_name, level=level,
                                   dry_run=dry_run, verbose=verbose)
            total_renamed += renamed

    print(f"\nTotal renamed/removed: {total_renamed}")


def main():
    parser = argparse.ArgumentParser(
        description='Rename groups in HDF5 files (with data comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename a tree in all shots
  python rename_h5_groups.py 200000.h5 d3d D3D --level tree

  # If D3D already exists and data is identical, d3d will be removed
  # If D3D already exists and data differs, error is raised

  # Rename a tree in specific shot
  python rename_h5_groups.py 200000.h5 d3d D3D --level tree --shot 200000

  # Rename a shot
  python rename_h5_groups.py data.h5 200000 200001 --level shot

  # Batch rename from file
  python rename_h5_groups.py 200000.h5 --batch mapping.txt --level tree

  # Dry run (preview changes)
  python rename_h5_groups.py 200000.h5 d3d D3D --level tree --dry-run

Behavior when new name exists:
  - Compares data recursively (structure, values, attributes)
  - If identical: removes old name (consolidates duplicates)
  - If different: raises error to prevent data loss
        """
    )

    parser.add_argument('file', help='HDF5 file to modify')
    parser.add_argument('old_name', nargs='?', help='Current group name')
    parser.add_argument('new_name', nargs='?', help='New group name')
    parser.add_argument('--level', choices=['shot', 'tree', 'signal'],
                        default='tree',
                        help='What to rename (default: tree)')
    parser.add_argument('--shot', help='Shot number (for tree/signal level)')
    parser.add_argument('--tree', help='Tree name (for signal level)')
    parser.add_argument('--batch', help='Batch rename from mapping file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be renamed without doing it')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        batch_rename(args.file, args.batch, level=args.level,
                     dry_run=args.dry_run, verbose=not args.quiet)
    else:
        # Single rename mode
        if not args.old_name or not args.new_name:
            parser.error("old_name and new_name required (or use --batch)")

        rename_group(args.file, args.old_name, args.new_name,
                     level=args.level, shot=args.shot, tree=args.tree,
                     dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == '__main__':
    main()
