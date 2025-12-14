"""Check for duplicate chunks"""
import json
from collections import Counter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

# Get config and setup path
config = get_config()
project_root = Path(__file__).parent.parent.parent
input_file = project_root / config.summaries_output_file

chunk_ids = []
total_lines = 0

print("Analyzing chunks...")

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        chunk = json.loads(line)
        chunk_id = chunk.get('chunk_id', f"unknown_{total_lines}")
        chunk_ids.append(chunk_id)

print(f"\n{'='*60}")
print("CHUNK ANALYSIS")
print('='*60)
print(f"Total lines: {total_lines:,}")
print(f"Unique chunk IDs: {len(set(chunk_ids)):,}")
print(f"Duplicates: {total_lines - len(set(chunk_ids)):,}")

# Find most duplicated
counter = Counter(chunk_ids)
duplicates = {k: v for k, v in counter.items() if v > 1}

if duplicates:
    print(f"\n⚠️ Found {len(duplicates)} chunk IDs with duplicates")
    print("\nTop 5 most duplicated:")
    for chunk_id, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {chunk_id}: {count} copies")
else:
    print("\n✅ No duplicates found!")

print('='*60)
