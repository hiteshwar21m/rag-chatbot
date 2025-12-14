"""
Production-Grade Docling PDF Extractor - OPTIMIZED VERSION
- Multi-threaded processing (2-4x faster)
- Better table extraction with cell matching
- Explicit OCR configuration
- Language support
- Preserves folder structure
- Resume capability (survives interruptions)
- Section-by-section processing
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions,
    AcceleratorOptions
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from pathlib import Path
import json
import time
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/extraction_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedExtractor:
    def __init__(
        self,
        input_root: str,
        output_root: str,
        checkpoint_file: str = "extraction_checkpoint_optimized.json",
        ocr_enabled: bool = True,
        output_format: str = "markdown",
        num_threads: int = 4,
        ocr_languages: list = ["en"]
    ):
        """
        Initialize optimized extractor.
        
        Args:
            input_root: Root directory (e.g., 'manit_pdfs')
            output_root: Output root (e.g., 'manit_extracted_optimized')
            checkpoint_file: File to track progress
            ocr_enabled: Enable OCR for scanned docs
            output_format: markdown, json, html, or text
            num_threads: Number of CPU threads (4-8 for i7)
            ocr_languages: OCR languages ["en"], ["hi"], or ["en", "hi"]
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.checkpoint_file = Path(checkpoint_file)
        self.ocr_enabled = ocr_enabled
        self.output_format = output_format
        self.num_threads = num_threads
        self.ocr_languages = ocr_languages
        
        # Create output root
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize checkpoint
        self.checkpoint = self._load_checkpoint()
        
        # Statistics
        self.stats = {
            "total_found": 0,
            "already_extracted": 0,
            "newly_extracted": 0,
            "failed": 0,
            "skipped_empty": 0,
            "start_time": datetime.now()
        }
        
        logger.info("="*60)
        logger.info("OPTIMIZED Production Extractor Initialized")
        logger.info("="*60)
        logger.info(f"Input root:      {self.input_root}")
        logger.info(f"Output root:     {self.output_root}")
        logger.info(f"Checkpoint:      {self.checkpoint_file}")
        logger.info(f"OCR enabled:     {self.ocr_enabled}")
        logger.info(f"OCR languages:   {', '.join(self.ocr_languages)}")
        logger.info(f"CPU threads:     {self.num_threads}")
        logger.info(f"Format:          {self.output_format}")
        logger.info(f"Optimizations:   Multi-threading, Cell matching, Explicit OCR")
        logger.info("="*60)
    
    def _load_checkpoint(self) -> dict:
        """Load checkpoint file if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"✓ Loaded checkpoint: {len(checkpoint.get('completed', []))} files already processed")
            return checkpoint
        else:
            logger.info("✓ No checkpoint found, starting fresh")
            return {"completed": [], "failed": [], "last_updated": None}
    
    def _save_checkpoint(self):
        """Save current progress."""
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate unique hash for file (path-based)."""
        rel_path = str(file_path.relative_to(self.input_root))
        return hashlib.md5(rel_path.encode()).hexdigest()
    
    def _is_already_processed(self, pdf_path: Path) -> bool:
        """Check if file was already successfully extracted."""
        file_hash = self._get_file_hash(pdf_path)
        return file_hash in self.checkpoint.get("completed", [])
    
    def _get_output_path(self, pdf_path: Path) -> Path:
        """Get mirrored output path preserving folder structure."""
        rel_path = pdf_path.relative_to(self.input_root)
        
        ext_map = {
            "markdown": ".md",
            "json": ".json",
            "html": ".html",
            "text": ".txt"
        }
        ext = ext_map.get(self.output_format, ".md")
        
        output_path = self.output_root / rel_path.parent / f"{rel_path.stem}{ext}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def _extract_single_pdf(self, pdf_path: Path) -> dict:
        """Extract content from a single PDF with optimized settings."""
        result = {
            "file": str(pdf_path),
            "success": False,
            "error": None,
            "processing_time": 0,
            "output_file": None,
            "content_length": 0,
            "skipped": False
        }
        
        try:
            # Check if already processed
            if self._is_already_processed(pdf_path):
                result["success"] = True
                result["skipped"] = True
                result["output_file"] = str(self._get_output_path(pdf_path))
                return result
            
            output_path = self._get_output_path(pdf_path)
            start_time = time.time()
            
            # OPTIMIZED PIPELINE CONFIGURATION
            pipeline_options = PdfPipelineOptions()
            
            # 1. Explicit OCR enable
            pipeline_options.do_ocr = self.ocr_enabled
            
            # 2. Table extraction with cell matching (better quality)
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            # 3. OCR options with language specification
            if self.ocr_enabled:
                pipeline_options.ocr_options = EasyOcrOptions(
                    force_full_page_ocr=False,
                    lang=self.ocr_languages
                )
            
            # 4. CRITICAL: Multi-threading for speed (2-4x faster)
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=self.num_threads,
                device=AcceleratorDevice.AUTO  # Auto-detect GPU/CPU
            )
            
            # Initialize converter with optimized options
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            
            # Convert
            conv_result = converter.convert(str(pdf_path))
            
            # Export based on format
            if self.output_format == "markdown":
                content = conv_result.document.export_to_markdown()
            elif self.output_format == "json":
                content = json.dumps(conv_result.document.export_to_dict(), indent=2)
            elif self.output_format == "html":
                content = conv_result.document.export_to_html()
            elif self.output_format == "text":
                content = conv_result.document.export_to_text()
            else:
                content = conv_result.document.export_to_markdown()
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result["success"] = True
            result["processing_time"] = time.time() - start_time
            result["output_file"] = str(output_path)
            result["content_length"] = len(content)
            
            # Mark as completed in checkpoint
            file_hash = self._get_file_hash(pdf_path)
            if file_hash not in self.checkpoint.get("completed", []):
                self.checkpoint["completed"].append(file_hash)
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            
            # Mark as failed
            file_hash = self._get_file_hash(pdf_path)
            if file_hash not in self.checkpoint.get("failed", []):
                self.checkpoint["failed"].append(file_hash)
        
        return result
    
    def scan_directory_structure(self) -> dict:
        """Scan and return directory structure with PDF counts."""
        structure = {}
        
        for section_dir in self.input_root.iterdir():
            if not section_dir.is_dir():
                continue
            
            section_name = section_dir.name
            structure[section_name] = {
                "path": str(section_dir),
                "url_folders": {},
                "total_pdfs": 0
            }
            
            for url_folder in section_dir.rglob("*"):
                if url_folder.is_dir():
                    pdf_count = len(list(url_folder.glob("*.pdf")))
                    if pdf_count > 0:
                        rel_path = str(url_folder.relative_to(section_dir))
                        structure[section_name]["url_folders"][rel_path] = pdf_count
                        structure[section_name]["total_pdfs"] += pdf_count
        
        return structure
    
    def extract_section(self, section_name: str):
        """Extract all PDFs from a specific section."""
        section_path = self.input_root / section_name
        
        if not section_path.exists():
            logger.error(f"Section not found: {section_name}")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Section: {section_name}")
        logger.info(f"{'='*60}")
        
        pdf_files = list(section_path.rglob("*.pdf"))
        
        if len(pdf_files) == 0:
            logger.warning(f"No PDFs found in section: {section_name}")
            self.stats["skipped_empty"] += 1
            return
        
        logger.info(f"Found {len(pdf_files)} PDFs in section '{section_name}'")
        self.stats["total_found"] += len(pdf_files)
        
        # Process each PDF
        for pdf_file in tqdm(pdf_files, desc=f"Extracting {section_name}"):
            result = self._extract_single_pdf(pdf_file)
            
            if result.get("skipped"):
                self.stats["already_extracted"] += 1
            elif result["success"]:
                self.stats["newly_extracted"] += 1
            else:
                self.stats["failed"] += 1
            
            # Save checkpoint every 10 files
            if (self.stats["newly_extracted"] + self.stats["failed"]) % 10 == 0:
                self._save_checkpoint()
        
        self._save_checkpoint()
        logger.info(f"✓ Completed section: {section_name}")
    
    def extract_all(self):
        """Extract all sections."""
        logger.info("\n" + "="*60)
        logger.info("EXTRACTING ALL SECTIONS (OPTIMIZED)")
        logger.info("="*60)
        
        sections = [d.name for d in self.input_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(sections)} sections: {', '.join(sections)}")
        
        for section in sections:
            try:
                self.extract_section(section)
            except KeyboardInterrupt:
                logger.warning("\n⚠️ Interrupted by user. Progress saved!")
                self._save_checkpoint()
                raise
            except Exception as e:
                logger.error(f"Error processing section {section}: {str(e)}")
                continue
        
        self._save_checkpoint()
    
    def print_summary(self):
        """Print extraction summary."""
        duration = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        print("\n" + "="*60)
        print("OPTIMIZED EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total PDFs found:       {self.stats['total_found']}")
        print(f"Already extracted:      {self.stats['already_extracted']}")
        print(f"Newly extracted:        {self.stats['newly_extracted']}")
        print(f"Failed:                 {self.stats['failed']}")
        print(f"Empty sections skipped: {self.stats['skipped_empty']}")
        print(f"Total time:             {duration:.2f}s ({duration/60:.2f} min)")
        
        if self.stats['newly_extracted'] > 0:
            avg_time = duration / self.stats['newly_extracted']
            print(f"Avg time per PDF:       {avg_time:.2f}s")
            
            # Compare to non-optimized
            estimated_speedup = 78 / avg_time  # 78s was baseline from tests
            print(f"Speed improvement:      ~{estimated_speedup:.1f}x faster")
        
        success_rate = (self.stats['newly_extracted'] / self.stats['total_found'] * 100) if self.stats['total_found'] > 0 else 0
        print(f"Success rate:           {success_rate:.1f}%")
        print("="*60)
        
        if self.stats['failed'] > 0:
            print(f"\n⚠️ {self.stats['failed']} files failed. Check log for details.")
        
        print(f"\n✓ Checkpoint saved: {self.checkpoint_file}")
        print(f"✓ Output directory: {self.output_root}")


def main():
    """Main entry point."""
    
    # ============ OPTIMIZED CONFIGURATION ============
    INPUT_ROOT = "data/raw/pdfs"
    OUTPUT_ROOT = "data/extracted/pdf_markdown"
    CHECKPOINT_FILE = "logs/extraction_checkpoint_optimized.json"
    
    # OCR Settings
    OCR_ENABLED = True
    OCR_LANGUAGES = ["en", "hi"]  # Change to ["en", "hi"] if Hindi content
    
    # Performance Settings
    NUM_THREADS = 4  # Adjust based on your CPU:
                     # i7 8th gen+: try 6 or 8
                     # i5 or older i7: use 4
                     # i3: use 2
    
    OUTPUT_FORMAT = "markdown"  # markdown, json, html, or text
    # ================================================
    
    print("\n" + "="*60)
    print("OPTIMIZED EXTRACTOR - Configuration")
    print("="*60)
    print(f"CPU Threads:     {NUM_THREADS}")
    print(f"OCR Languages:   {', '.join(OCR_LANGUAGES)}")
    print(f"Optimizations:   ✓ Multi-threading")
    print(f"                 ✓ Cell matching for tables")
    print(f"                 ✓ Explicit OCR configuration")
    print(f"Expected Speed:  2-4x faster than basic version")
    print("="*60)
    
    # Initialize extractor
    extractor = OptimizedExtractor(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        checkpoint_file=CHECKPOINT_FILE,
        ocr_enabled=OCR_ENABLED,
        output_format=OUTPUT_FORMAT,
        num_threads=NUM_THREADS,
        ocr_languages=OCR_LANGUAGES
    )
    
    # Show directory structure
    print("\n" + "="*60)
    print("SCANNING DIRECTORY STRUCTURE")
    print("="*60)
    structure = extractor.scan_directory_structure()
    
    total_pdfs = 0
    for section, info in structure.items():
        print(f"\n📁 {section}/ ({info['total_pdfs']} PDFs)")
        for url_folder, count in sorted(info['url_folders'].items()):
            print(f"   └─ {url_folder}/ ({count} PDFs)")
        total_pdfs += info['total_pdfs']
    
    print(f"\n📊 Total PDFs across all sections: {total_pdfs}")
    print("="*60)
    
    # Ask user what to extract
    print("\nOptions:")
    print("1. Extract ALL sections (optimized)")
    print("2. Extract specific section (optimized)")
    print("3. Show structure only (no extraction)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    try:
        if choice == "1":
            extractor.extract_all()
        
        elif choice == "2":
            print("\nAvailable sections:")
            for i, section in enumerate(structure.keys(), 1):
                print(f"{i}. {section} ({structure[section]['total_pdfs']} PDFs)")
            
            section_choice = input("\nEnter section name: ").strip()
            if section_choice in structure:
                extractor.extract_section(section_choice)
            else:
                print(f"❌ Section '{section_choice}' not found!")
        
        elif choice == "3":
            print("\n✓ Structure displayed above. No extraction performed.")
            return
        
        else:
            print("❌ Invalid choice!")
            return
        
        extractor.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Extraction interrupted!")
        print("✓ Progress has been saved.")
        print("✓ Run the script again to resume.")
        extractor.print_summary()


if __name__ == "__main__":
    main()