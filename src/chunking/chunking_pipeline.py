"""
RAG-Ready Chunking Pipeline for MANIT Documents
Processes both PDF markdown files and webpage text files with layout-aware chunking.
Optimized for hierarchical retrieval with document summaries.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from tqdm import tqdm
import tiktoken
import sys

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

try:
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DoclingDocument, TextItem, TableItem
except ImportError:
    print("Installing docling packages...")
    import subprocess
    subprocess.run(["pip", "install", "docling", "docling-core"], check=True)
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DoclingDocument, TextItem, TableItem


@dataclass
class ChunkMetadata:
    """RAG-optimized chunk metadata"""
    chunk_id: str
    text: str
    document_id: str
    document_title: str
    section: str
    subsection: Optional[str]
    source_type: str  # 'pdf' or 'webpage'
    file_path: str
    chunk_index: int
    total_chunks: int
    has_table: bool
    has_heading: bool
    heading_text: Optional[str]
    estimated_tokens: int


class RAGChunker:
    """Production-ready chunker optimized for RAG retrieval"""
    
    def __init__(
        self,
        pdf_dir: str = None,
        webpage_dir: str = None,
        output_file: str = None,
        max_tokens: int = 512,
        min_tokens: int = 64,
        overlap_tokens: int = 128,
        model_name: str = None
    ):
        config = get_config()
        if model_name is None:
            model_name = config.embedding_model
        if pdf_dir is None:
            pdf_dir = config.pdf_markdown_dir
        if webpage_dir is None:
            webpage_dir = config.webpage_text_dir
        if output_file is None:
            output_file = config.chunks_output_file
        
        self.pdf_dir = Path(pdf_dir)
        self.webpage_dir = Path(webpage_dir)
        self.output_file = output_file
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Docling HybridChunker config
        self.chunker = HybridChunker(
            tokenizer=model_name,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            merge_peers=True  # Combine small adjacent sections
        )
        
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        self.stats = {
            'total_files': 0,
            'total_chunks': 0,
            'pdf_files': 0,
            'webpage_files': 0,
            'skipped_files': 0,
            'errors': []
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID from file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    def extract_title_from_markdown(self, content: str) -> Optional[str]:
        """Extract first ## heading as document title"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                # Remove ## and clean up
                title = line[3:].strip()
                # Remove markdown formatting
                title = re.sub(r'\*\*|__|\*|_|`', '', title)
                return title[:200]  # Limit title length
        return None
    
    def extract_title_from_filename(self, filename: str) -> str:
        """Extract readable title from filename"""
        # Remove extension
        name = Path(filename).stem
        
        # For webpages: clean_www-manit-ac-in_about-us -> About Us
        if name.startswith('clean_www-manit-ac-in_'):
            name = name.replace('clean_www-manit-ac-in_', '')
        
        # Replace separators with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Remove special characters and clean up
        name = re.sub(r'[%\d]+', '', name)
        name = ' '.join(name.split())  # Remove extra spaces
        
        # Title case
        return name.title()[:200]
    
    def clean_markdown_content(self, content: str) -> str:
        """Clean markdown content for chunking"""
        # Remove image placeholders
        content = re.sub(r'<!--\s*image\s*-->', '', content, flags=re.IGNORECASE)
        
        # Remove multiple empty lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def detect_table_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect markdown table boundaries (start, end line numbers)"""
        lines = text.split('\n')
        tables = []
        in_table = False
        table_start = 0
        
        for i, line in enumerate(lines):
            # Markdown table line contains |
            is_table_line = '|' in line and line.strip().startswith('|')
            
            if is_table_line and not in_table:
                in_table = True
                table_start = i
            elif not is_table_line and in_table:
                in_table = False
                tables.append((table_start, i - 1))
        
        # Handle table at end of file
        if in_table:
            tables.append((table_start, len(lines) - 1))
        
        return tables
    
    def extract_heading_before_chunk(self, content: str, chunk_start: int) -> Optional[str]:
        """Extract the most recent heading before chunk position"""
        text_before = content[:chunk_start]
        lines = text_before.split('\n')
        
        # Look backwards for last heading
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('##'):
                heading = line.lstrip('#').strip()
                # Clean markdown formatting
                heading = re.sub(r'\*\*|__|\*|_|`', '', heading)
                return heading[:150]
        
        return None
    
    def chunk_markdown_content(
        self,
        content: str,
        file_path: str,
        section: str,
        subsection: Optional[str]
    ) -> List[Dict]:
        """Chunk markdown content with table awareness"""
        
        # Clean content
        content = self.clean_markdown_content(content)
        
        if not content.strip():
            return []
        
        # Extract document title
        doc_title = self.extract_title_from_markdown(content)
        if not doc_title:
            doc_title = self.extract_title_from_filename(file_path)
        
        doc_id = self.generate_doc_id(file_path)
        
        # Detect tables
        table_boundaries = self.detect_table_boundaries(content)
        
        # Simple chunking strategy: split by double newlines and headings
        chunks_text = []
        current_chunk = []
        current_tokens = 0
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if we're in a table
            in_table = False
            for table_start, table_end in table_boundaries:
                if table_start <= i <= table_end:
                    in_table = True
                    # Extract entire table
                    table_lines = lines[table_start:table_end + 1]
                    table_text = '\n'.join(table_lines)
                    
                    # If current chunk + table is too big, save current chunk
                    if current_chunk:
                        chunks_text.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    
                    # Add table as its own chunk (even if large)
                    chunks_text.append(table_text)
                    
                    i = table_end + 1
                    break
            
            if in_table:
                continue
            
            # Check if heading
            is_heading = line.strip().startswith('#')
            
            # Estimate tokens
            line_tokens = self.count_tokens(line)
            
            # If adding this line exceeds max_tokens, save current chunk
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunks_text.append('\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
                
                # Start new chunk with heading if applicable
                if is_heading:
                    current_chunk.append(line)
                    current_tokens = line_tokens
                    i += 1
                    continue
            
            # Add line to current chunk
            if line.strip():  # Don't add empty lines at chunk start
                current_chunk.append(line)
                current_tokens += line_tokens
            
            i += 1
        
        # Add remaining chunk
        if current_chunk:
            chunks_text.append('\n'.join(current_chunk))
        
        # Create chunk metadata
        chunks = []
        total_chunks = len(chunks_text)
        
        for idx, chunk_text in enumerate(chunks_text):
            # Extract heading for this chunk
            chunk_start_pos = content.find(chunk_text)
            heading = self.extract_heading_before_chunk(content, chunk_start_pos)
            
            # Check if chunk has table
            has_table = '|' in chunk_text and chunk_text.count('|') > 5
            
            # Check if chunk has heading
            has_heading = any(line.strip().startswith('#') for line in chunk_text.split('\n'))
            
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{doc_id}_chunk_{idx:03d}",
                text=chunk_text.strip(),
                document_id=doc_id,
                document_title=doc_title,
                section=section,
                subsection=subsection,
                source_type='pdf',
                file_path=str(file_path),
                chunk_index=idx,
                total_chunks=total_chunks,
                has_table=has_table,
                has_heading=has_heading,
                heading_text=heading,
                estimated_tokens=self.count_tokens(chunk_text)
            )
            
            chunks.append(asdict(chunk_metadata))
        
        return chunks
    
    def chunk_webpage_content(
        self,
        content: str,
        file_path: str,
        section: str
    ) -> List[Dict]:
        """Chunk webpage text content"""
        
        content = content.strip()
        
        # Skip tiny files (likely just links or empty)
        if len(content) < 50:
            self.stats['skipped_files'] += 1
            return []
        
        # Extract title from filename
        doc_title = self.extract_title_from_filename(file_path)
        doc_id = self.generate_doc_id(file_path)
        
        # Simple paragraph-based chunking for webpages
        paragraphs = content.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks_text = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If paragraph alone exceeds max, split by sentences
            if para_tokens > self.max_tokens:
                if current_chunk:
                    chunks_text.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long paragraph
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                        chunks_text.append(' '.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            else:
                # Normal paragraph chunking
                if current_tokens + para_tokens > self.max_tokens and current_chunk:
                    chunks_text.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining
        if current_chunk:
            chunks_text.append('\n'.join(current_chunk))
        
        # Create metadata
        chunks = []
        total_chunks = len(chunks_text)
        
        for idx, chunk_text in enumerate(chunks_text):
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{doc_id}_chunk_{idx:03d}",
                text=chunk_text.strip(),
                document_id=doc_id,
                document_title=doc_title,
                section=section,
                subsection=None,
                source_type='webpage',
                file_path=str(file_path),
                chunk_index=idx,
                total_chunks=total_chunks,
                has_table=False,
                has_heading=False,
                heading_text=None,
                estimated_tokens=self.count_tokens(chunk_text)
            )
            
            chunks.append(asdict(chunk_metadata))
        
        return chunks
    
    def process_pdf_file(self, file_path: Path) -> List[Dict]:
        """Process a single PDF markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract section and subsection from path
            parts = file_path.relative_to(self.pdf_dir).parts
            section = parts[0] if len(parts) > 0 else "unknown"
            subsection = parts[1] if len(parts) > 2 else None
            
            chunks = self.chunk_markdown_content(
                content=content,
                file_path=str(file_path),
                section=section,
                subsection=subsection
            )
            
            self.stats['pdf_files'] += 1
            self.stats['total_chunks'] += len(chunks)
            
            return chunks
            
        except Exception as e:
            self.stats['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })
            return []
    
    def process_webpage_file(self, file_path: Path) -> List[Dict]:
        """Process a single webpage text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract section from filename
            filename = file_path.stem
            if filename.startswith('clean_www-manit-ac-in_'):
                section_part = filename.replace('clean_www-manit-ac-in_', '')
                # Take first meaningful part
                section = section_part.split('_')[0] if '_' in section_part else section_part
            else:
                section = "general"
            
            chunks = self.chunk_webpage_content(
                content=content,
                file_path=str(file_path),
                section=section
            )
            
            if chunks:  # Only count if not skipped
                self.stats['webpage_files'] += 1
                self.stats['total_chunks'] += len(chunks)
            
            return chunks
            
        except Exception as e:
            self.stats['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })
            return []
    
    def process_all(self):
        """Process all PDF and webpage files"""
        all_chunks = []
        
        print("🚀 Starting RAG-Ready Chunking Pipeline...")
        print(f"📂 PDF directory: {self.pdf_dir}")
        print(f"📄 Webpage directory: {self.webpage_dir}")
        print(f"📊 Max tokens per chunk: {self.max_tokens}")
        print()
        
        # Process PDFs
        print("📚 Processing PDF markdown files...")
        pdf_files = list(self.pdf_dir.rglob("*.md"))
        
        for pdf_file in tqdm(pdf_files, desc="PDF files"):
            chunks = self.process_pdf_file(pdf_file)
            all_chunks.extend(chunks)
            self.stats['total_files'] += 1
        
        # Process webpages
        print("\n🌐 Processing webpage text files...")
        webpage_files = list(self.webpage_dir.glob("*.txt"))
        
        for webpage_file in tqdm(webpage_files, desc="Webpage files"):
            chunks = self.process_webpage_file(webpage_file)
            all_chunks.extend(chunks)
            self.stats['total_files'] += 1
        
        # Write output
        print(f"\n💾 Writing chunks to {self.output_file}...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Print stats
        print("\n" + "="*60)
        print("✅ CHUNKING COMPLETE")
        print("="*60)
        print(f"📊 Total files processed: {self.stats['total_files']}")
        print(f"   - PDF files: {self.stats['pdf_files']}")
        print(f"   - Webpage files: {self.stats['webpage_files']}")
        print(f"   - Skipped (too small): {self.stats['skipped_files']}")
        print(f"\n📦 Total chunks created: {self.stats['total_chunks']}")
        print(f"📁 Output file: {self.output_file}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            print("First 5 errors:")
            for err in self.stats['errors'][:5]:
                print(f"  - {err['file']}: {err['error']}")
        
        print("\n🎯 Chunks are RAG-ready with:")
        print("   ✓ Layout-aware chunking")
        print("   ✓ Table preservation")
        print("   ✓ Metadata for filtering")
        print("   ✓ Citation-ready paths")
        print("   ✓ Heading context")
        print("="*60)


def main():
    """Main entry point"""
    # Get project root (two levels up from src/chunking/)
    project_root = Path(__file__).parent.parent.parent
    
    chunker = RAGChunker(
        max_tokens=512,
        min_tokens=64,
        overlap_tokens=128
    )
    
    chunker.process_all()


if __name__ == "__main__":
    main()
