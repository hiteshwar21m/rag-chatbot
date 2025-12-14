#!/usr/bin/env python3
"""
PDF Download Diagnostic Script
Analyzes the downloaded PDFs and identifies issues with the download process.
"""

import os
import json
from urllib.parse import urlparse
from collections import defaultdict, Counter
import hashlib

DOWNLOAD_DIR = "manit_pdfs"

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def analyze_pdfs():
    """Analyze the downloaded PDF files."""
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Directory {DOWNLOAD_DIR} does not exist!")
        return
    
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith('.pdf')]
    
    print(f"=== PDF Download Analysis ===")
    print(f"Actual files in directory: {len(files)}")
    print(f"Directory: {os.path.abspath(DOWNLOAD_DIR)}")
    print()
    
    if len(files) == 0:
        print("No PDF files found!")
        return
    
    # Analyze file sizes
    file_sizes = []
    zero_byte_files = []
    small_files = []  # Less than 1KB
    duplicate_hashes = defaultdict(list)
    
    for filename in files:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        try:
            size = os.path.getsize(filepath)
            file_sizes.append(size)
            
            if size == 0:
                zero_byte_files.append(filename)
            elif size < 1024:  # Less than 1KB
                small_files.append((filename, size))
            
            # Check for duplicates by hash
            file_hash = get_file_hash(filepath)
            if file_hash:
                duplicate_hashes[file_hash].append(filename)
                
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    print(f"=== File Size Analysis ===")
    print(f"Total files: {len(files)}")
    print(f"Zero-byte files: {len(zero_byte_files)}")
    print(f"Files smaller than 1KB: {len(small_files)}")
    if file_sizes:
        print(f"Average file size: {sum(file_sizes)/len(file_sizes):.0f} bytes")
        print(f"Largest file: {max(file_sizes)} bytes")
        print(f"Smallest non-zero file: {min([s for s in file_sizes if s > 0]) if any(s > 0 for s in file_sizes) else 0} bytes")
    print()
    
    # Show zero-byte files
    if zero_byte_files:
        print(f"=== Zero-byte Files (first 10) ===")
        for filename in zero_byte_files[:10]:
            print(f"  {filename}")
        if len(zero_byte_files) > 10:
            print(f"  ... and {len(zero_byte_files) - 10} more")
        print()
    
    # Show small files
    if small_files:
        print(f"=== Small Files (<1KB, first 10) ===")
        for filename, size in small_files[:10]:
            print(f"  {filename}: {size} bytes")
        if len(small_files) > 10:
            print(f"  ... and {len(small_files) - 10} more")
        print()
    
    # Check for duplicates
    duplicates = {hash_val: filenames for hash_val, filenames in duplicate_hashes.items() if len(filenames) > 1}
    if duplicates:
        print(f"=== Duplicate Files (same content) ===")
        print(f"Number of duplicate groups: {len(duplicates)}")
        total_duplicates = sum(len(filenames) - 1 for filenames in duplicates.values())
        print(f"Total duplicate files: {total_duplicates}")
        
        # Show first few duplicate groups
        for i, (hash_val, filenames) in enumerate(list(duplicates.items())[:5]):
            print(f"  Group {i+1} ({len(filenames)} files):")
            for filename in filenames[:3]:
                print(f"    {filename}")
            if len(filenames) > 3:
                print(f"    ... and {len(filenames) - 3} more")
        print()
    
    # Analyze filename patterns
    print(f"=== Filename Pattern Analysis ===")
    filename_counter = Counter()
    url_patterns = Counter()
    
    for filename in files:
        # Count similar filenames (might indicate overwrites)
        base_name = filename.replace('.pdf', '')
        filename_counter[base_name] += 1
        
        # Analyze URL patterns if possible
        if 'sites_default_files' in filename:
            url_patterns['sites/default/files'] += 1
        elif 'documents' in filename:
            url_patterns['documents'] += 1
        else:
            url_patterns['other'] += 1
    
    print("URL pattern distribution:")
    for pattern, count in url_patterns.most_common():
        print(f"  {pattern}: {count} files")
    print()
    
    # Check for potential overwrites
    potential_overwrites = {name: count for name, count in filename_counter.items() if count > 1}
    if potential_overwrites:
        print(f"=== Potential Filename Conflicts ===")
        print(f"Files with same base name: {len(potential_overwrites)}")
        for name, count in list(potential_overwrites.items())[:10]:
            print(f"  {name}: {count} instances")
        print()
    
    # Summary and recommendations
    print(f"=== Summary and Recommendations ===")
    unique_files = len(files) - total_duplicates if 'duplicates' in locals() else len(files)
    valid_files = len([f for f in file_sizes if f > 1024])  # Files larger than 1KB
    
    print(f"Actual unique files: {unique_files}")
    print(f"Files likely to be valid PDFs: {valid_files}")
    
    if zero_byte_files:
        print(f"⚠️  {len(zero_byte_files)} zero-byte files suggest download failures")
    if small_files:
        print(f"⚠️  {len(small_files)} very small files might be error pages or incomplete downloads")
    if 'duplicates' in locals() and duplicates:
        print(f"ℹ️  {total_duplicates} duplicate files can be removed to save space")
    
    discrepancy = 4853 - len(files)
    if discrepancy > 0:
        print(f"📊 Discrepancy analysis:")
        print(f"   Logged downloads: 4853")
        print(f"   Actual files: {len(files)}")
        print(f"   Missing files: {discrepancy}")
        print(f"   Possible causes:")
        print(f"   - File overwrites due to duplicate names")
        print(f"   - Failed downloads not properly logged")
        print(f"   - Files downloaded to wrong location")
        print(f"   - Download process interrupted and resumed")

def check_log_file():
    """Analyze the log file for additional insights."""
    log_file = "pdf_download.log"
    if not os.path.exists(log_file):
        print("No log file found!")
        return
    
    print(f"\n=== Log File Analysis ===")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Count different types of log messages
        download_success = log_content.count("Downloaded ")
        download_skip = log_content.count("Skipping ")
        download_error = log_content.count("Error downloading")
        strategy_failures = log_content.count("All download strategies failed")
        
        print(f"Successful downloads logged: {download_success}")
        print(f"Skipped (already exists): {download_skip}")
        print(f"Download errors: {download_error}")
        print(f"Complete strategy failures: {strategy_failures}")
        
        # Check for specific error patterns
        if "certificate verify failed" in log_content:
            print("⚠️  SSL certificate errors detected")
        if "Connection timeout" in log_content:
            print("⚠️  Connection timeout errors detected")
        if "Max retries exceeded" in log_content:
            print("⚠️  Max retry errors detected")
            
    except Exception as e:
        print(f"Error reading log file: {e}")

def main():
    analyze_pdfs()
    check_log_file()
    
    print(f"\n=== Next Steps ===")
    print("1. Check if there are multiple download directories")
    print("2. Look for any .tmp or partial download files")
    print("3. Consider re-running the download with debug logging")
    print("4. Check if Windows is hiding file extensions")

if __name__ == "__main__":
    main()