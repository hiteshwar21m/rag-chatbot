"""
Extraction Manager - Helper Script
Quick commands to manage your extraction process
"""

from pathlib import Path
import json
from datetime import datetime

def load_checkpoint(checkpoint_file="logs/extraction_checkpoint.json"):
    """Load checkpoint data."""
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def show_progress():
    """Show current extraction progress."""
    checkpoint = load_checkpoint()
    
    if not checkpoint:
        print("❌ No checkpoint found. No extraction started yet.")
        return
    
    completed = len(checkpoint.get("completed", []))
    failed = len(checkpoint.get("failed", []))
    last_updated = checkpoint.get("last_updated", "Unknown")
    
    print("\n" + "="*60)
    print("CURRENT EXTRACTION PROGRESS")
    print("="*60)
    print(f"✓ Successfully extracted: {completed} files")
    print(f"✗ Failed:                 {failed} files")
    print(f"📅 Last updated:          {last_updated}")
    print("="*60)

def reset_checkpoint():
    """Reset checkpoint to start fresh."""
    checkpoint_file = Path("logs/extraction_checkpoint.json")
    
    if checkpoint_file.exists():
        # Backup existing checkpoint
        backup_name = f"logs/extraction_checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        checkpoint_file.rename(backup_name)
        print(f"✓ Backed up existing checkpoint to: {backup_name}")
    
    print("✓ Checkpoint reset. Next run will start fresh.")

def count_pdfs(root_dir="data/raw/pdfs"):
    """Count PDFs in directory structure."""
    root = Path(root_dir)
    
    if not root.exists():
        print(f"❌ Directory not found: {root_dir}")
        return
    
    print("\n" + "="*60)
    print("PDF COUNT BY SECTION")
    print("="*60)
    
    total = 0
    for section in sorted(root.iterdir()):
        if section.is_dir():
            count = len(list(section.rglob("*.pdf")))
            print(f"📁 {section.name:25} {count:5} PDFs")
            total += count
    
    print("="*60)
    print(f"📊 TOTAL:                     {total:5} PDFs")
    print("="*60)

def verify_output(input_root="data/raw/pdfs", output_root="data/extracted/pdf_markdown"):
    """Verify extraction output matches input structure."""
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_root}")
        return
    
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_root}")
        return
    
    print("\n" + "="*60)
    print("VERIFICATION: Input vs Output")
    print("="*60)
    
    total_input = 0
    total_output = 0
    
    for section in sorted(input_path.iterdir()):
        if section.is_dir():
            input_count = len(list(section.rglob("*.pdf")))
            
            output_section = output_path / section.name
            output_count = 0
            if output_section.exists():
                # Count markdown/json/html/txt files
                output_count = len(list(output_section.rglob("*.md")))
                output_count += len(list(output_section.rglob("*.json")))
                output_count += len(list(output_section.rglob("*.html")))
                output_count += len(list(output_section.rglob("*.txt")))
            
            status = "✓" if input_count == output_count else "⚠"
            print(f"{status} {section.name:20} Input: {input_count:4} | Output: {output_count:4}")
            
            total_input += input_count
            total_output += output_count
    
    print("="*60)
    print(f"   {'TOTAL':20} Input: {total_input:4} | Output: {total_output:4}")
    print("="*60)
    
    if total_input == total_output:
        print("\n✓ All files extracted successfully!")
    else:
        missing = total_input - total_output
        print(f"\n⚠ {missing} files not yet extracted")

def list_failed():
    """List files that failed extraction."""
    checkpoint = load_checkpoint()
    
    if not checkpoint:
        print("❌ No checkpoint found.")
        return
    
    failed = checkpoint.get("failed", [])
    
    if not failed:
        print("✓ No failed extractions!")
        return
    
    print("\n" + "="*60)
    print(f"FAILED EXTRACTIONS ({len(failed)} files)")
    print("="*60)
    print("\nThese file hashes failed:")
    for hash_val in failed:
        print(f"  - {hash_val}")
    print("\n💡 Check extraction log for details")

def main():
    """Main menu."""
    print("\n" + "="*60)
    print("EXTRACTION MANAGER")
    print("="*60)
    print("\n1. Show extraction progress")
    print("2. Count PDFs by section")
    print("3. Verify extraction (compare input vs output)")
    print("4. List failed extractions")
    print("5. Reset checkpoint (start fresh)")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        show_progress()
    elif choice == "2":
        count_pdfs()
    elif choice == "3":
        verify_output()
    elif choice == "4":
        list_failed()
    elif choice == "5":
        confirm = input("⚠ This will reset all progress. Continue? (yes/no): ")
        if confirm.lower() == "yes":
            reset_checkpoint()
    elif choice == "6":
        print("Goodbye!")
        return
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()