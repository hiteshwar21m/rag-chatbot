"""
Extract document summaries and upload to Weaviate
Creates a separate collection for document-level search
"""
import json
import weaviate
import weaviate.classes as wvc
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

def extract_document_summaries():
    """Extract unique documents with summaries"""
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / config.chunks_file
    
    print("📂 Extracting document summaries...")
    docs = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            doc_id = chunk.get('document_id')
            
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    'document_id': doc_id,
                    'document_title': chunk.get('document_title', ''),
                    'document_summary': chunk.get('document_summary', ''),
                    'source_type': chunk.get('source_type', ''),
                }
    
    print(f"✅ Found {len(docs)} unique documents")
    return list(docs.values())


def upload_to_weaviate(documents):
    """Upload document summaries to Weaviate"""
    
    print("\n🔄 Initializing...")
    config = get_config()
    client = weaviate.connect_to_local(
        host=config.weaviate_host,
        port=config.weaviate_http_port,
        grpc_port=config.weaviate_grpc_port
    )
    model = SentenceTransformer(config.embedding_model)
    
    try:
        # Delete collection if exists
        print("Creating collection...")
        if client.collections.exists("ManitDocumentSummary"):
            client.collections.delete("ManitDocumentSummary")
            print("🗑️  Deleted existing collection")
        
        # Create collection
        client.collections.create(
            name="ManitDocumentSummary",
            description="Document-level summaries for hierarchical retrieval",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=[
                wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
                wvc.config.Property(name="document_title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="document_summary", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_type", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            ]
        )
        print("✅ Collection created!")
        
        # Upload documents
        print("\n📤 Uploading documents...")
        collection = client.collections.get("ManitDocumentSummary")
        uploaded = 0
        
        with collection.batch.dynamic() as batch:
            for doc in tqdm(documents, desc="Uploading"):
                # Generate embedding of summary
                summary_text = doc['document_summary'] or doc['document_title']
                vector = model.encode(summary_text).tolist()
                
                # Upload
                batch.add_object(
                    properties={
                        "document_id": doc['document_id'],
                        "document_title": doc['document_title'],
                        "document_summary": doc['document_summary'],
                        "source_type": doc['source_type'],
                    },
                    vector=vector
                )
                uploaded += 1
        
        # Verify
        print("\n✅ Verifying...")
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count
        
        print("\n" + "=" * 70)
        print("✅ UPLOAD COMPLETE!")
        print("=" * 70)
        print(f"Uploaded: {uploaded:,} documents")
        print(f"In Weaviate: {count:,} documents")
        print("=" * 70)
    
    finally:
        client.close()


def main():
    print("=" * 70)
    print("🚀 DOCUMENT SUMMARY COLLECTION")
    print("=" * 70)
    
    # Extract
    documents = extract_document_summaries()
    
    # Upload
    upload_to_weaviate(documents)


if __name__ == "__main__":
    main()
