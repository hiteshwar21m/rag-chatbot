"""
Upload chunks with embeddings to Weaviate (v4 API) - Simplified
"""
import json
import weaviate
import weaviate.classes as wvc
from pathlib import Path
from tqdm import tqdm
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

def main():
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / config.embeddings_file
    
    print("=" * 70)
    print("🚀 WEAVIATE UPLOAD")
    print("=" * 70)
    
    # Connect to Weaviate v4 - with retry
    print("\n1️⃣ Connecting to Weaviate...")
    max_retries = 5
    for i in range(max_retries):
        try:
            client = weaviate.connect_to_local(
                host=config.weaviate_host,
                port=config.weaviate_http_port,
                grpc_port=config.weaviate_grpc_port,
            )
            if client.is_ready():
                print("✅ Connected!")
                break
        except Exception as e:
            if i < max_retries - 1:
                print(f"   Retry {i+1}/{max_retries}... (waiting 5s)")
                time.sleep(5)
            else:
                print(f"❌ Failed to connect: {e}")
                return
    
    try:
        # Delete collection if exists
        print("\n2️⃣ Creating schema...")
        if client.collections.exists("ManitChunk"):
            client.collections.delete("ManitChunk")
            print("🗑️  Deleted existing collection")
        
        # Create collection
        client.collections.create(
            name="ManitChunk",
            description="RAG chunks from MANIT documents",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
                wvc.config.Property(name="document_title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="document_summary", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="section", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_type", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="token_count", data_type=wvc.config.DataType.INT),
            ]
        )
        print("✅ Collection created!")
        
        # Count chunks
        print("\n3️⃣ Counting chunks...")
        total = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        print(f"✅ Found {total:,} chunks")
        
        # Upload in batches
        print("\n4️⃣ Uploading to Weaviate...")
        collection = client.collections.get("ManitChunk")
        uploaded = 0
        
        with collection.batch.dynamic() as batch:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total, desc="Uploading"):
                    chunk = json.loads(line)
                    
                    # Prepare object
                    properties = {
                        "text": chunk.get('text', ''),
                        "document_id": chunk.get('document_id', ''),
                        "document_title": chunk.get('document_title', ''),
                        "document_summary": chunk.get('document_summary') or '',
                        "section": chunk.get('section', ''),
                        "source_type": chunk.get('source_type', ''),
                        "chunk_index": chunk.get('chunk_index', 0),
                        "token_count": chunk.get('token_count', 0),
                    }
                    
                    vector = chunk.get('embedding')
                    
                    # Add to batch
                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
                    
                    uploaded += 1
        
        # Verify
        print("\n5️⃣ Verifying upload...")
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count
        
        print("\n" + "=" * 70)
        print("✅ UPLOAD COMPLETE!")
        print("=" * 70)
        print(f"Uploaded: {uploaded:,} chunks")
        print(f"In Weaviate: {count:,} chunks")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()


if __name__ == "__main__":
    main()
