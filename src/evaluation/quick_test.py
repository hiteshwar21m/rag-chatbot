"""
Quick Evaluation - Test hybrid retrieval on 200 random queries
"""
import json
import sys
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent / \"retrieval\"))
from hybrid_retriever import HybridRetriever


def extract_queries(sample_size=200):
    """Extract random sample of queries"""
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data/processed/chunks_final.jsonl"
    
    queries_by_doc = defaultdict(list)
    
    print("📂 Extracting sample queries...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            doc_id = chunk.get('document_id')
            queries = chunk.get('sample_queries', [])
            
            if doc_id and queries:
                if doc_id not in queries_by_doc:
                    queries_by_doc[doc_id] = queries
    
    all_queries = []
    for doc_id, queries in queries_by_doc.items():
        for query in queries:
            all_queries.append({
                'query': query,
                'expected_doc_id': doc_id
            })
    
    # Random sample
    sample = random.sample(all_queries, min(sample_size, len(all_queries)))
    print(f"✅ Sampled {len(sample)} queries from {len(all_queries)} total")
    return sample


def evaluate(queries):
    """Evaluate hybrid retrieval"""
    
    print("\n🔄 Initializing hybrid retriever...")
    retriever = HybridRetriever()
    
    print(f"Evaluating {len(queries)} queries...")
    
    correct = 0
    total_mrr = 0.0
    
    for item in tqdm(queries, desc="Evaluating"):
        query_text = item['query']
        expected_doc_id = item['expected_doc_id']
        
        try:
            results = retriever.retrieve(query_text, top_k=10)
            result_doc_ids = [chunk['document_id'] for chunk in results]
            
            if expected_doc_id in result_doc_ids:
                correct += 1
                rank = result_doc_ids.index(expected_doc_id) + 1
                total_mrr += 1.0 / rank
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    retriever.close()
    
    accuracy = correct / len(queries)
    mrr = total_mrr / len(queries)
    
    return {
        'total': len(queries),
        'correct': correct,
        'accuracy': accuracy,
        'mrr': mrr
    }


def main():
    random.seed(42)  # Reproducible
    
    print("=" * 70)
    print("🎯 QUICK HYBRID RETRIEVAL EVALUATION (100 queries)")
    print("=" * 70)
    
    queries = extract_queries(sample_size=100)
    results = evaluate(queries)
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Queries Tested: {results['total']}")
    print(f"Correct Retrievals: {results['correct']}")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print(f"MRR: {results['mrr']:.3f}")
    print("=" * 70)
    
    # Compare
    simple_accuracy = 44.0
    hybrid_accuracy = results['accuracy'] * 100
    improvement = hybrid_accuracy - simple_accuracy
    
    print("\n📈 COMPARISON:")
    print(f"Simple Retrieval:  {simple_accuracy:.1f}%")
    print(f"Hybrid Retrieval:  {hybrid_accuracy:.1f}%")
    print(f"Improvement:       {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n✅ Hybrid is {improvement:.1f}% better!")
    else:
        print(f"\n⚠️ Hybrid is {abs(improvement):.1f}% worse (unexpected)")


if __name__ == "__main__":
    main()
