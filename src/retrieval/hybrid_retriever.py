"""
Hybrid Retrieval Engine
Combines summary-based and chunk-based search with cross-encoder reranking
"""
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

class HybridRetriever:
    """Two-path retrieval with cross-encoder reranking"""
    
    def __init__(self):
        config = get_config()
        self.client = weaviate.connect_to_local()
        self.model = SentenceTransformer(config.embedding_model)
        self.reranker = CrossEncoder(config.reranker_model)
        self.chunk_collection = self.client.collections.get("ManitChunk")
        self.doc_collection = self.client.collections.get("ManitDocumentSummary")
    
    def search_summaries(self, query: str, top_k: int = 10) -> List[str]:
        """Path A: Search document summaries, return document IDs"""
        
        query_vector = self.model.encode(query).tolist()
        
        response = self.doc_collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["document_id"]
        )
        
        return [obj.properties['document_id'] for obj in response.objects]
    
    def get_chunks_by_doc_ids(self, doc_ids: List[str]) -> List[Dict]:
        """Get all chunks from specific documents"""
        
        chunks = []
        
        for doc_id in doc_ids:
            response = self.chunk_collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("document_id").equal(doc_id),
                limit=1000,  # Max chunks per doc
                return_properties=["text", "document_id", "document_title", "section", "chunk_index"]
            )
            
            for obj in response.objects:
                chunks.append({
                    **obj.properties,
                    'source': 'path_a',
                    'distance': 0.0  # Will be reranked later
                })
        
        return chunks
    
    def search_chunks(self, query: str, top_k: int = 20) -> List[Dict]:
        """Path B: Direct chunk search"""
        
        query_vector = self.model.encode(query).tolist()
        
        response = self.chunk_collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["text", "document_id", "document_title", "section", "chunk_index"],
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                **obj.properties,
                'source': 'path_b',
                'distance': obj.metadata.distance
            })
        
        return chunks
    
    def merge_results(self, path_a_chunks: List[Dict], path_b_chunks: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results from both paths"""
        
        # Use dict to deduplicate (key = document_id + chunk_index)
        merged = {}
        
        for chunk in path_a_chunks + path_b_chunks:
            key = f"{chunk['document_id']}_{chunk.get('chunk_index', 0)}"
            if key not in merged:
                merged[key] = chunk
        
        return list(merged.values())
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank merged results using cross-encoder"""
        
        # Prepare query-chunk pairs for cross-encoder
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get relevance scores from cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['cross_encoder_score'] = float(score)
            
            # Bonus for chunks from summary-based search
            doc_bonus = 0.1 if chunk['source'] == 'path_a' else 0
            
            # Combined score
            chunk['final_score'] = score + doc_bonus
        
        # Sort by score
        sorted_chunks = sorted(chunks, key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity: max 2 chunks per document
        final_results = []
        doc_count = defaultdict(int)
        
        for chunk in sorted_chunks:
            doc_id = chunk['document_id']
            if doc_count[doc_id] < 2:  # Max 2 per doc
                final_results.append(chunk)
                doc_count[doc_id] += 1
            
            if len(final_results) >= top_k:
                break
        
        return final_results
    
    def retrieve(self, query: str, top_k: int = 5, logger=None) -> List[Dict]:
        """
        Main retrieval method with optional logging
        Returns top-k most relevant chunks using hybrid search
        """
        import time
        
        # Path A: Summary-based search
        start = time.time()
        relevant_doc_ids = self.search_summaries(query, top_k=10)
        path_a_chunks = self.get_chunks_by_doc_ids(relevant_doc_ids)
        path_a_time = time.time() - start
        
        if logger:
            logger.log_path_a(path_a_time, relevant_doc_ids, len(path_a_chunks))
        
        # Path B: Direct chunk search
        start = time.time()
        path_b_chunks = self.search_chunks(query, top_k=20)
        path_b_time = time.time() - start
        
        if logger:
            chunk_ids = [f"{c['document_id']}_{c.get('chunk_index', 0)}" for c in path_b_chunks]
            logger.log_path_b(path_b_time, chunk_ids, len(path_b_chunks))
        
        # Merge
        start = time.time()
        all_chunks = self.merge_results(path_a_chunks, path_b_chunks)
        merge_time = time.time() - start
        
        if logger:
            logger.log_merge(merge_time, len(path_a_chunks) + len(path_b_chunks), len(all_chunks))
        
        # Rerank
        start = time.time()
        final_chunks = self.rerank(query, all_chunks, top_k=top_k)
        rerank_time = time.time() - start
        
        if logger:
            final_ids = [f"{c['document_id']}_{c.get('chunk_index', 0)}" for c in final_chunks]
            logger.log_reranking(rerank_time, final_ids)
        
        return final_chunks
    
    def close(self):
        """Close Weaviate connection"""
        self.client.close()


# Example usage
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    query = "What are the B.Tech admission requirements?"
    results = retriever.retrieve(query, top_k=5)
    
    print(f"Query: {query}\n")
    print("="*70)
    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. {chunk['document_title']}")
        print(f"   Section: {chunk['section']}")
        print(f"   Score: {chunk['final_score']:.3f}")
        print(f"   Source: {chunk['source']}")
        print(f"   Text: {chunk['text'][:150]}...")
        print("-"*70)
    
    retriever.close()
