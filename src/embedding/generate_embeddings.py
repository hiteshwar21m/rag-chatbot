"""
Memory-Efficient Streaming Embedding Pipeline
Processes chunks one-by-one without loading all into memory
"""
import json
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

def count_lines(file_path):
    """Count total lines for progress bar"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def main():
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / config.chunks_file
    output_file = project_root / config.embeddings_file
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer(config.embedding_model, device=device)
    print(f"✅ Model loaded! Dimension: {model.get_sentence_embedding_dimension()}")
    
    # Count total for progress
    print("\nCounting chunks...")
    total = count_lines(input_file)
    print(f"✅ Found {total:,} chunks to process")
    
    # Ensure output dir exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in streaming fashion with batching
    print("\n🔄 Generating embeddings...")
    batch_size = config.embedding_batch_size
    batch_texts = []
    batch_chunks = []
    processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8') as outf:
        
        for line in tqdm(inf, total=total, desc="Progress"):
            chunk = json.loads(line)
            
            # Skip if no text
            if 'text' not in chunk or not chunk['text']:
                continue
            
            batch_texts.append(chunk['text'])
            batch_chunks.append(chunk)
            
            # Process when batch is full
            if len(batch_texts) >= batch_size:
                # Generate embeddings
                embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                ).tolist()
                
                # Write immediately
                for ch, emb in zip(batch_chunks, embeddings):
                    ch['embedding'] = emb
                    outf.write(json.dumps(ch, ensure_ascii=False) + '\n')
                
                processed += len(batch_texts)
                
                # Clear batch
                batch_texts = []
                batch_chunks = []
        
        # Process remaining
        if batch_texts:
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            for ch, emb in zip(batch_chunks, embeddings):
                ch['embedding'] = emb
                outf.write(json.dumps(ch, ensure_ascii=False) + '\n')
            
            processed += len(batch_texts)
    
    # Done
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"Processed: {processed:,} chunks")
    print(f"Output: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
