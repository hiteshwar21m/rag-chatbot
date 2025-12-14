"""
Evaluate Hybrid Retrieval System
Compare hybrid vs simple retrieval on sample queries
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add path to import hybrid retriever
sys.path.append(str(Path(__file__).parent.parent / \"retrieval\"))
from hybrid_retriever import HybridRetriever


def extract_queries():
    """Extract all sample queries with their document IDs"""
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
    
    print(f"✅ Extracted {len(all_queries)} queries from {len(queries_by_doc)} documents")
    return all_queries


def evaluate_hybrid(queries, top_k=10):
    """Evaluate hybrid retrieval"""
    
    print("\n🔄 Initializing hybrid retriever...")
    retriever = HybridRetriever()
    
    print(f"Running evaluation on {len(queries)} queries...")
    
    correct_retrievals = 0
    total_mrr = 0.0
    total_precision_at_5 = 0.0
    failed_queries = []
    
    for item in tqdm(queries, desc="Evaluating"):
        query_text = item['query']
        expected_doc_id = item['expected_doc_id']
        
        try:
            # Use hybrid retrieval
            results = retriever.retrieve(query_text, top_k=top_k)
            
            # Extract document IDs
            result_doc_ids = [chunk['document_id'] for chunk in results]
            
            # Check if expected doc appears
            if expected_doc_id in result_doc_ids:
                correct_retrievals += 1
                rank = result_doc_ids.index(expected_doc_id) + 1
                total_mrr += 1.0 / rank
                
                relevant_in_top5 = sum(1 for doc_id in result_doc_ids[:5] if doc_id == expected_doc_id)
                total_precision_at_5 += relevant_in_top5 / 5.0
            else:
                failed_queries.append({
                    'query': query_text,
                    'expected': expected_doc_id,
                    'got': result_doc_ids[:3]
                })
        
        except Exception as e:
            print(f"\n❌ Error on query '{query_text}': {e}")
            failed_queries.append({
                'query': query_text,
                'error': str(e)
            })
    
    retriever.close()
    
    # Calculate metrics
    total = len(queries)
    return {
        'total_queries': total,
        'correct_retrievals': correct_retrievals,
        'accuracy': correct_retrievals / total,
        'mrr': total_mrr / total,
        'precision_at_5': total_precision_at_5 / total,
        'failed_count': len(failed_queries),
        'failed_queries': failed_queries[:10]
    }


def main():
    print("=" * 70)
    print("🎯 HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)
    
    # Extract queries
    queries = extract_queries()
    
    # Evaluate
    results = evaluate_hybrid(queries, top_k=10)
    
    # Display
    print("\n" + "=" * 70)
    print("📊 HYBRID RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"Total Queries: {results['total_queries']:,}")
    print(f"Correct Retrievals: {results['correct_retrievals']:,}")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print(f"MRR: {results['mrr']:.3f}")
    print(f"Precision@5: {results['precision_at_5']:.3f}")
    print(f"Failed Queries: {results['failed_count']:,}")
    print("=" * 70)
    
    # Compare with simple retrieval
    simple_accuracy = 0.44
    improvement = (results['accuracy'] - simple_accuracy) / simple_accuracy * 100
    
    print("\n📈 COMPARISON:")
    print(f"Simple Retrieval: {simple_accuracy*100:.1f}%")
    print(f"Hybrid Retrieval: {results['accuracy']*100:.1f}%")
    print(f"Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"✅ Hybrid retrieval is {improvement:.1f}% better!")
    
    # Save report
    project_root = Path(__file__).parent.parent.parent
    report_file = project_root / "logs/hybrid_evaluation_report.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Report saved: {report_file}")


if __name__ == "__main__":
    main()
