"""
View Query Logs - Analyze RAG performance
"""
import json
from pathlib import Path
from query_logger import get_logger
import sys

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

def view_recent_logs(n=10):
    """View recent query logs"""
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / config.query_log_file
    
    if not log_file.exists():
        print("No logs found yet!")
        return
    
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            logs.append(json.loads(line))
    
    recent = logs[-n:]
    
    print(f"\n{'='*80}")
    print(f"RECENT {len(recent)} QUERIES")
    print('='*80)
    
    for i, log in enumerate(recent, 1):
        print(f"\n{i}. Query: {log['query']}")
        print(f"   Time: {log['timestamp']}")
        print(f"   Total: {log['total_time']}s")
        print(f"   Retrieval: {log['retrieval']['path_a']['time'] + log['retrieval']['path_b']['time'] + log['retrieval']['merge']['time'] + log['retrieval']['reranking']['time']}s")
        print(f"   LLM: {log['llm']['time']}s")
        print(f"   Path A: {log['retrieval']['path_a']['chunks_retrieved']} chunks from {len(log['retrieval']['path_a']['top_docs'])} docs")
        print(f"   Path B: {log['retrieval']['path_b']['chunks_retrieved']} chunks")
        print(f"   Final: {len(log['retrieval']['reranking']['final_chunks'])} chunks")
        print("-" * 80)


def view_stats():
    """View summary statistics"""
    logger = get_logger()
    stats = logger.get_summary_stats()
    
    if stats['total_queries'] == 0:
        print("No queries logged yet!")
        return
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Avg Total Time: {stats['avg_total_time']}s")
    print(f"Avg Retrieval Time: {stats['avg_retrieval_time']}s ({stats['retrieval_percentage']}%)")
    print(f"Avg LLM Time: {stats['avg_llm_time']}s ({stats['llm_percentage']}%)")
    print('='*80)


if __name__ == "__main__":
    view_recent_logs(5)
    view_stats()
