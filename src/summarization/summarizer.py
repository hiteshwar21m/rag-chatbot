"""
Optimized Summarization Pipeline - Uses Paid Gemini (Cheap & Fast) with Free Fallbacks
Cost: ~$0.98 for entire dataset (1,432 documents)
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set
from collections import defaultdict
from datetime import datetime
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "config" / ".env")


class IncrementalSummarizationPipeline:
    """Production-ready summarization with paid primary + free fallbacks"""
    
    def __init__(
        self,
        input_file: str = None,
        output_file: str = None,
        error_log: str = None,
        max_daily_calls: int = 2000,
        model: str = None, # Paid model as requested
        delay_between_calls: float = 1.0
    ):
        self.project_root = Path(__file__).parent.parent.parent
        # Get config
        config = get_config()

        # Set defaults from config if not provided
        if model is None:
            model = config.llm_model
        if input_file is None:
            input_file = config.chunks_file
        if output_file is None:
            output_file = config.summaries_output_file
        if error_log is None:
            error_log = config.summarization_error_log

        self.input_file = self.project_root / input_file
        self.output_file = self.project_root / output_file
        self.error_log = self.project_root / error_log
                
        self.max_daily_calls = max_daily_calls
        self.model = model
        self.delay_between_calls = delay_between_calls
        
        # OpenRouter API
        self.api_key = config.openrouter_api_key
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "MANIT Document Summarizer"
        }
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.api_calls_made = 0
        self.total_cost = 0.0
        
        # Stats
        self.model_usage = {
            "google/gemini-2.0-flash-lite-001": 0,
            "google/gemini-2.0-flash-exp:free": 0,
            "meta-llama/llama-3.3-70b-instruct:free": 0,
            "qwen/qwen-2.5-72b-instruct:free": 0
        }
        
        # Ensure dirs exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Gemini 1.5 Flash"""
        input_cost = (input_tokens / 1_000_000) * 0.075
        output_cost = (output_tokens / 1_000_000) * 0.30
        return input_cost + output_cost
    
    def log_error(self, doc_id: str, error_type: str, details: str):
        with open(self.error_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'document_id': doc_id,
                'error_type': error_type,
                'details': details
            }) + '\n')
    
    def call_api(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Smart API call with automatic fallback:
        1. Try paid Gemini 1.5 Flash (fast, stable, <$1 total)
        2. Fallback to free Gemini 2.0 if needed
        3. Fallback to free Llama 3.3 70B if needed
        """
        
        # Try models in priority order (verified OpenRouter IDs)
        model_priority = [
            self.model,                                      # Paid (primary)  
            "google/gemini-2.0-flash-exp:free",             # Free (fallback 1)
            "meta-llama/llama-3.3-70b-instruct:free",       # Free (fallback 2)
            "qwen/qwen-2.5-72b-instruct:free"               # Free (fallback 3)
        ]
        
        for model in model_priority:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            
            # Retry current model up to max_retries
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=120  # 2 minutes
                    )
                    
                    self.api_calls_made += 1
                    
                    # Success!
                    if response.status_code == 200:
                        result = response.json()
                        self.model_usage[model] = self.model_usage.get(model, 0) + 1
                        return {
                            'success': result['choices'][0]['message']['content'],
                            'model_used': model
                        }
                    
                    # Rate limit - try next model
                    elif response.status_code == 429:
                        print(f"  ⚠️ Rate limit on {model}, trying next model...")
                        break
                    
                    # Server error - retry
                    elif response.status_code >= 500:
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt
                            print(f"  ⚠️ Server error, retrying in {wait}s...")
                            time.sleep(wait)
                            continue
                        else:
                            break  # Try next model
                    
                    # Other error - try next model
                    else:
                        print(f"  ❌ Error {response.status_code} on {model}")
                        break
                
                except requests.exceptions.ReadTimeout:
                    if attempt < max_retries - 1:
                        print(f"  ⏱️ Timeout on {model}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"  ⏱️ Timeout on {model}, trying next model...")
                        break
                
                except Exception as e:
                    print(f"  ⚠️ Exception on {model}: {str(e)[:100]}")
                    break
            
            print(f"  ⚠️ Moving to fallback model...")
        
        # All models failed
        return {'error': 'All models failed after retries'}
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            response = response.strip()
            
            # Handle markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            response = response.strip()
            return json.loads(response)
        except:
            return None
    
    def summarize_document(self, doc_id: str, chunks: list, doc_title: str, section: str) -> Optional[Dict]:
        """Generate summary for one document"""
        
        # Combine chunks
        full_text = "\n\n".join(c['text'] for c in chunks)
        token_count = self.count_tokens(full_text)
        
        # Truncate if needed (save costs)
        if token_count > 30000:
            full_text = full_text[:100000]
            token_count = self.count_tokens(full_text)
        
        prompt = f"""You are an expert analyst. Analyze this document from MANIT Bhopal.

Document: {doc_title}
Section: {section}

Content:
{full_text}

Generate a JSON response with:
1. "summary": A clear, comprehensive summary (100-150 words) covering main topics, purpose, and key information
2. "queries": Array of exactly 3 diverse questions this document answers:
   - First: A factual question (What/Who/When/Where)
   - Second: A procedural question (How to do something)
   - Third: A specific detail or requirement question

Strict JSON format:
{{
  "summary": "Your 100-150 word summary here...",
  "queries": [
    "Specific factual question?",
    "How-to procedural question?",
    "Requirement or detail question?"
  ]
}}

Return ONLY valid JSON, no other text."""

        # Call API
        result = self.call_api(prompt)
        
        if not result or 'error' in result:
            self.log_error(doc_id, 'api_failed', str(result))
            return None
        
        # Track cost if paid model was used
        if result.get('model_used') == self.model:
            output_tokens = 150  # Approximate
            cost = self.estimate_cost(token_count, output_tokens)
            self.total_cost += cost
        
        # Parse response
        parsed = self.parse_json_response(result['success'])
        if not parsed:
            self.log_error(doc_id, 'json_parse_failed', result['success'][:500])
            return None
        
        return parsed
    
    def get_already_processed_docs(self) -> Set[str]:
        """Get documents that already have summaries"""
        processed = set()
        
        if not self.output_file.exists():
            return processed
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    if chunk.get('document_summary'):
                        processed.add(chunk['document_id'])
                except:
                    continue
        
        return processed
    
    def write_chunks_for_document(self, doc_id: str, all_chunks: Dict, summary_data: Optional[Dict]):
        """Write chunks for one document immediately"""
        doc_chunks = all_chunks[doc_id]
        
        for chunk in doc_chunks:
            if summary_data:
                chunk['document_summary'] = summary_data.get('summary', 'Summary generation failed')
                chunk['sample_queries'] = summary_data.get('queries', [])
            else:
                chunk['document_summary'] = None
                chunk['sample_queries'] = []
        
        # Append immediately
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for chunk in doc_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    def run(self):
        """Main pipeline execution"""
        
        print("=" * 70)
        print("🚀 PRODUCTION SUMMARIZATION PIPELINE")
        print("=" * 70)
        print(f"Primary Model: {self.model} (Paid)")
        print(f"Estimated Cost: ~$0.98 for 1,432 documents")
        print(f"Fallback Models: Free Gemini 2.0, Llama 3.3 70B")
        print()
        
        # Load chunks
        print("📂 Loading chunks...")
        chunks_by_doc = defaultdict(list)
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks_by_doc[chunk['document_id']].append(chunk)
        
        # Get already processed
        already_processed = self.get_already_processed_docs()
        
        # Filter remaining
        remaining = {
            k: v for k, v in chunks_by_doc.items() 
            if k not in already_processed
        }
        
        print(f"📋 Total documents: {len(chunks_by_doc)}")
        print(f"✅ Already processed: {len(already_processed)}")
        print(f"⏳ To process: {len(remaining)}")
        print("=" * 70)
        print()
        
        # Process documents
        success_count = 0
        error_count = 0
        
        for doc_id, chunks in tqdm(list(remaining.items()), desc="Summarizing"):
            # Check limit
            if self.api_calls_made >= self.max_daily_calls:
                print("\n🛑 Daily limit reached")
                break
            
            # Get metadata
            doc_title = chunks[0]['document_title']
            section = chunks[0]['section']
            
            # Summarize
            summary_data = self.summarize_document(doc_id, chunks, doc_title, section)
            
            # Write immediately
            self.write_chunks_for_document(doc_id, {doc_id: chunks}, summary_data)
            
            if summary_data:
                success_count += 1
            else:
                error_count += 1
            
            # Rate limiting
            time.sleep(self.delay_between_calls)
        
        # Final stats
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"✅ Successfully processed: {success_count}")
        print(f"❌ Errors: {error_count}")
        print(f"📞 API calls made: {self.api_calls_made}")
        print(f"💰 Estimated cost: ${self.total_cost:.2f}")
        print()
        print("🤖 Model usage:")
        for model, count in self.model_usage.items():
            if count > 0:
                print(f"   - {model}: {count} calls")
        print()
        print(f"📁 Output: {self.output_file}")
        
        if error_count > 0:
            print(f"📝 Error log: {self.error_log}")
        
        if self.api_calls_made >= self.max_daily_calls:
            print(f"\n⏳ Remaining: {len(remaining) - success_count - error_count} documents")
            print("💡 Run again to continue!")
        
        print("=" * 70)


def main():
    pipeline = IncrementalSummarizationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
