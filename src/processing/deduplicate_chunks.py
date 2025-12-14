"""
Deduplicate chunks and merge summaries
Takes original chunks + adds summaries from summarized file
"""
import json
from collections import defaultdict
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

# Get configuration
config = get_config()
project_root = Path(__file__).parent.parent.parent

original_file = project_root / "data/processed/chunks_with_metadata.jsonl"
summary_file = project_root / config.summaries_output_file
output_file = project_root / config.chunks_file

print("=" * 70)
print("DEDUPLICATION & SUMMARY MERGE")
print("=" * 70)

# Step 1: Load summaries by document_id
print("\n1️⃣ Loading summaries...")
summaries_by_doc = {}

with open(summary_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Reading summaries"):
        chunk = json.loads(line)
        doc_id = chunk.get('document_id')
        if doc_id and 'document_summary' in chunk:
            # Store first occurrence of summary for each doc
            if doc_id not in summaries_by_doc:
                summaries_by_doc[doc_id] = {
                    'summary': chunk.get('document_summary'),
                    'queries': chunk.get('sample_queries', [])
                }

print(f"✅ Found summaries for {len(summaries_by_doc):,} documents")

# Step 2: Merge with original chunks
print("\n2️⃣ Merging with original chunks...")
processed = 0
with_summary = 0
without_summary = 0

with open(original_file, 'r', encoding='utf-8') as inf, \
     open(output_file, 'w', encoding='utf-8') as outf:
    
    for line in tqdm(inf, desc="Processing"):
        chunk = json.loads(line)
        doc_id = chunk.get('document_id')
        
        # Add summary if available
        if doc_id in summaries_by_doc:
            chunk['document_summary'] = summaries_by_doc[doc_id]['summary']
            chunk['sample_queries'] = summaries_by_doc[doc_id]['queries']
            with_summary += 1
        else:
            chunk['document_summary'] = None
            chunk['sample_queries'] = []
            without_summary += 1
        
        outf.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        processed += 1

# Results
print("\n" + "=" * 70)
print("✅ DEDUPLICATION COMPLETE!")
print("=" * 70)
print(f"Total chunks written:      {processed:,}")
print(f"  With summaries:          {with_summary:,}")
print(f"  Without summaries:       {without_summary:,}")
print(f"\nOutput: {output_file}")
print("=" * 70)

print("\n💡 Next steps:")
print("   1. Delete chunks_with_summaries.jsonl (has duplicates)")
print("   2. Use chunks_final.jsonl for embeddings")
