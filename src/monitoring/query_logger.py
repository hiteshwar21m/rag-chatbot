"""
Query Logger - Track RAG pipeline performance
Logs every query with detailed metrics for optimization
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
from pathlib import Path as PathObj

sys.path.append(str(PathObj(__file__).parent.parent / "config"))
from settings import get_config

class QueryLogger:
    """Logs query execution details for debugging and optimization"""
    
    def __init__(self, log_file: str = None):
        config = get_config()
        if log_file is None:
            log_file = config.query_log_file
        self.project_root = Path(__file__).parent.parent.parent
        self.log_file = self.project_root / log_file
        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        
        self.current_query = None
        self.start_time = None
    
    def start_query(self, query: str):
        """Start logging a new query"""
        self.current_query = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "retrieval": {
                "path_a": {},
                "path_b": {},
                "merge": {},
                "reranking": {}
            },
            "llm": {},
            "total_time": 0
        }
        self.start_time = time.time()
    
    def log_path_a(self, time_taken: float, top_docs: List[str], chunks_retrieved: int):
        """Log Path A (summary-based) results"""
        self.current_query["retrieval"]["path_a"] = {
            "time": round(time_taken, 3),
            "top_docs": top_docs,
            "chunks_retrieved": chunks_retrieved
        }
    
    def log_path_b(self, time_taken: float, top_chunks: List[str], chunks_retrieved: int):
        """Log Path B (direct chunk) results"""
        self.current_query["retrieval"]["path_b"] = {
            "time": round(time_taken, 3),
            "top_chunks": top_chunks[:5],  # Only first 5 for brevity
            "chunks_retrieved": chunks_retrieved
        }
    
    def log_merge(self, time_taken: float, total_candidates: int, unique_chunks: int):
        """Log merge step"""
        self.current_query["retrieval"]["merge"] = {
            "time": round(time_taken, 3),
            "total_candidates": total_candidates,
            "unique_chunks": unique_chunks
        }
    
    def log_reranking(self, time_taken: float, final_chunk_ids: List[str]):
        """Log reranking results"""
        self.current_query["retrieval"]["reranking"] = {
            "time": round(time_taken, 3),
            "final_chunks": final_chunk_ids
        }
    
    def log_llm(self, time_taken: float, model: str, temperature: float, answer_length: int):
        """Log LLM call"""
        self.current_query["llm"] = {
            "time": round(time_taken, 3),
            "model": model,
            "temperature": temperature,
            "answer_length": answer_length
        }
    
    def end_query(self):
        """Finalize and save query log"""
        if self.current_query and self.start_time:
            self.current_query["total_time"] = round(time.time() - self.start_time, 3)
            
            # Write to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(self.current_query, ensure_ascii=False) + '\n')
            
            # Reset
            self.current_query = None
            self.start_time = None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all logs"""
        if not self.log_file.exists():
            return {"total_queries": 0}
        
        queries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line))
        
        if not queries:
            return {"total_queries": 0}
        
        total = len(queries)
        avg_total_time = sum(q["total_time"] for q in queries) / total
        avg_retrieval_time = sum(
            q["retrieval"]["path_a"].get("time", 0) +
            q["retrieval"]["path_b"].get("time", 0) +
            q["retrieval"]["merge"].get("time", 0) +
            q["retrieval"]["reranking"].get("time", 0)
            for q in queries
        ) / total
        avg_llm_time = sum(q["llm"].get("time", 0) for q in queries) / total
        
        return {
            "total_queries": total,
            "avg_total_time": round(avg_total_time, 3),
            "avg_retrieval_time": round(avg_retrieval_time, 3),
            "avg_llm_time": round(avg_llm_time, 3),
            "retrieval_percentage": round((avg_retrieval_time / avg_total_time) * 100, 1),
            "llm_percentage": round((avg_llm_time / avg_total_time) * 100, 1)
        }


# Singleton instance
_logger_instance = None

def get_logger() -> QueryLogger:
    """Get or create logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = QueryLogger()
    return _logger_instance
