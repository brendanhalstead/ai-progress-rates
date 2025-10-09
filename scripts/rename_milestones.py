#!/usr/bin/env python3
"""
Script to rename milestones in all rollouts.jsonl files.

Renames:
- "(Expensive, threshold only considers taste) SAR" -> "Top-human-experiment-selection-skill"
- "(Expensive, threshold only considers taste) SIAR" -> "SIAR-level-experiment-selection-skill"
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Milestone rename mappings
MILESTONE_RENAMES = {
    "(Expensive, threshold only considers taste) SAR": "Top-human-experiment-selection-skill",
    "(Expensive, threshold only considers taste) SIAR": "SIAR-level-experiment-selection-skill",
}


def rename_milestones_in_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rename milestones in a single rollout record."""
    results = record.get("results")
    if not isinstance(results, dict):
        return record

    milestones = results.get("milestones")
    if not isinstance(milestones, dict):
        return record

    # Create new milestones dict with renamed keys
    new_milestones = {}
    for old_name, milestone_data in milestones.items():
        new_name = MILESTONE_RENAMES.get(old_name, old_name)
        new_milestones[new_name] = milestone_data

    # Update the record
    results["milestones"] = new_milestones
    return record


def process_rollouts_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single rollouts.jsonl file."""
    print(f"Processing {filepath}")

    # Read all records
    records = []
    num_changed = 0

    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                records.append("")
                continue

            try:
                record = json.loads(line)

                # Check if any milestones need renaming
                results = record.get("results")
                if isinstance(results, dict):
                    milestones = results.get("milestones")
                    if isinstance(milestones, dict):
                        for old_name in MILESTONE_RENAMES.keys():
                            if old_name in milestones:
                                num_changed += 1
                                break

                # Rename milestones
                record = rename_milestones_in_record(record)
                records.append(json.dumps(record))

            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} has invalid JSON: {e}")
                records.append(line)

    if num_changed > 0:
        print(f"  Found {num_changed} records with old milestone names")

        if not dry_run:
            # Write back to file
            with filepath.open("w", encoding="utf-8") as f:
                for record_line in records:
                    f.write(record_line + "\n")
            print(f"  Updated {filepath}")
        else:
            print(f"  (dry run - no changes made)")
    else:
        print(f"  No changes needed")

    return num_changed


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Rename milestones in rollouts.jsonl files")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    parser.add_argument("paths", nargs="*", help="Specific paths to process (default: all outputs/**/rollouts.jsonl)")
    args = parser.parse_args()

    # Find all rollouts.jsonl files
    if args.paths:
        rollouts_files = [Path(p) for p in args.paths]
    else:
        repo_root = Path(__file__).parent.parent
        outputs_dir = repo_root / "outputs"
        if not outputs_dir.exists():
            print(f"Error: outputs directory not found: {outputs_dir}")
            sys.exit(1)

        rollouts_files = list(outputs_dir.glob("**/rollouts.jsonl"))

    if not rollouts_files:
        print("No rollouts.jsonl files found")
        return

    print(f"Found {len(rollouts_files)} rollouts.jsonl files")
    if args.dry_run:
        print("DRY RUN MODE - no changes will be made")
    print()

    total_changed = 0
    for filepath in rollouts_files:
        try:
            num_changed = process_rollouts_file(filepath, dry_run=args.dry_run)
            total_changed += num_changed
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

    print()
    print(f"Total records with old milestone names: {total_changed}")
    if args.dry_run:
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
