#!/usr/bin/env python3
"""
MANIT Website PDF Downloader
Downloads PDFs from MANIT website URLs with robust error handling and timeout management.
Organizes downloads in hierarchical folder structure: category/url_folder/pdfs
Only extracts PDFs from main content area, avoiding duplicate downloads from navigation.
"""

import requests
import json
import os
import time
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pathlib import Path
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import hashlib

# Disable SSL warnings when VERIFY_SSL is False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
DOWNLOAD_DIR = "manit_pdfs"
URLS_FILE = "urls.txt"
TRACKING_FILE = "download_tracking.json"
MAX_RETRIES = 2
BACKOFF_FACTOR = 1
TIMEOUT = 60
DELAY_BETWEEN_REQUESTS = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
VERIFY_SSL = False
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 90
DRY_RUN = False # Set to True to preview without downloading

# Set up logging with UTF-8 encoding support
log_handlers = [
    logging.FileHandler('pdf_download.log', encoding='utf-8')
]

# Add StreamHandler with error handling for Windows console
try:
    import sys
    # Try to set console to UTF-8 mode
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    log_handlers.append(logging.StreamHandler())
except:
    # Fallback: use StreamHandler with error='replace' to substitute problematic chars
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, errors='replace')
    log_handlers.append(stream_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

class DownloadTracker:
    """Track downloaded PDFs to avoid duplicates."""
    
    def __init__(self, tracking_file):
        self.tracking_file = tracking_file
        self.downloaded_pdfs = self.load_tracking()
    
    def load_tracking(self):
        """Load tracking data from file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load tracking file: {e}")
                return {}
        return {}
    
    def save_tracking(self):
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.downloaded_pdfs, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save tracking file: {e}")
    
    def get_pdf_hash(self, pdf_url):
        """Generate a unique hash for a PDF URL."""
        return hashlib.md5(pdf_url.encode()).hexdigest()
    
    def is_downloaded(self, pdf_url):
        """Check if PDF has already been downloaded."""
        pdf_hash = self.get_pdf_hash(pdf_url)
        return pdf_hash in self.downloaded_pdfs
    
    def mark_downloaded(self, pdf_url, file_path, source_url):
        """Mark a PDF as downloaded."""
        pdf_hash = self.get_pdf_hash(pdf_url)
        self.downloaded_pdfs[pdf_hash] = {
            'url': pdf_url,
            'file_path': file_path,
            'source_url': source_url,
            'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_tracking()
    
    def get_download_location(self, pdf_url):
        """Get the location where a PDF was previously downloaded."""
        pdf_hash = self.get_pdf_hash(pdf_url)
        if pdf_hash in self.downloaded_pdfs:
            return self.downloaded_pdfs[pdf_hash].get('file_path')
        return None

def create_session():
    """Create a requests session with retry strategy and proper headers."""
    session = requests.Session()
    
    # Retry strategy - compatible with different urllib3 versions
    try:
        retry_strategy = Retry(
            total=MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=BACKOFF_FACTOR,
            raise_on_status=False
        )
    except TypeError:
        retry_strategy = Retry(
            total=MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=BACKOFF_FACTOR,
            raise_on_status=False
        )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    })
    
    return session

def download_with_fallback(session, url, verify_ssl=VERIFY_SSL):
    """Download with multiple fallback strategies."""
    strategies = [
        {'url': url.replace('http://', 'https://'), 'timeout': (CONNECT_TIMEOUT, READ_TIMEOUT), 'verify': verify_ssl},
        {'url': url.replace('https://', 'http://'), 'timeout': (CONNECT_TIMEOUT, READ_TIMEOUT), 'verify': False},
        {'url': url, 'timeout': (60, 120), 'verify': False},
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            logging.debug(f"Trying strategy {i} for {strategy['url']}")
            response = session.get(strategy['url'], timeout=strategy['timeout'], 
                                 verify=strategy['verify'], stream=True, allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as e:
            logging.warning(f"Strategy {i} failed: {e}")
            if i < len(strategies):
                time.sleep(2)
            continue
    
    return None

def sanitize_filename(filename):
    """Sanitize filename for safe saving."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    return filename

def create_url_folder_name(url):
    """Create a folder name from URL path."""
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    
    if not path:
        folder_name = parsed_url.netloc.replace('.', '_')
    else:
        parts = path.split('/')
        if len(parts) <= 2:
            folder_name = '_'.join(parts)
        else:
            folder_name = parts[-1]
    
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
    folder_name = folder_name[:100]
    
    return folder_name if folder_name else 'default'

def find_pdf_links(session, url):
    """Find all PDF links on a webpage, ONLY from main content area."""
    pdf_links = []
    
    try:
        logging.info(f"Searching for PDFs on: {url}")
        response = download_with_fallback(session, url)
        if not response:
            logging.error(f"All strategies failed for {url}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main content area
        main_content = None
        
        # Primary selector: inner-page-content
        main_content = soup.find('section', class_='inner-page-content')
        
        if main_content:
            logging.info("✓ Found main content area: <section class='inner-page-content'>")
        else:
            # Fallback selectors if the primary one doesn't exist
            fallback_selectors = [
                ('main', {}),
                ('article', {}),
                ('div', {'class': 'content'}),
                ('div', {'id': 'content'}),
                ('div', {'class': 'main-content'}),
            ]
            
            for tag, attrs in fallback_selectors:
                main_content = soup.find(tag, attrs)
                if main_content:
                    logging.info(f"✓ Found main content area: <{tag} {attrs}>")
                    break
        
        if not main_content:
            logging.warning("⚠ Could not identify main content area! Skipping this page to avoid navigation PDFs.")
            logging.warning("  If this page should have PDFs, please check the HTML structure.")
            return []
        
        # Find all links ONLY within the main content area
        links = main_content.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = urljoin(url, href)
            elif not href.startswith(('http://', 'https://')):
                href = urljoin(url, href)
            
            # Check if link points to a PDF
            if href.lower().endswith('.pdf'):
                pdf_links.append(href)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pdf_links = []
        for link in pdf_links:
            if link not in seen:
                seen.add(link)
                unique_pdf_links.append(link)
        
        if unique_pdf_links:
            logging.info(f"✓ Found {len(unique_pdf_links)} unique PDF link(s) in main content")
        else:
            logging.info("○ No PDF links found in main content area")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing {url}: {e}")
    
    return unique_pdf_links

def download_pdf(session, pdf_url, download_dir, tracker, source_url):
    """Download a single PDF file with duplicate tracking."""
    try:
        # Check if already downloaded globally
        if tracker.is_downloaded(pdf_url):
            existing_location = tracker.get_download_location(pdf_url)
            logging.info(f"⊘ Skipping {pdf_url}")
            logging.info(f"  Already downloaded to: {existing_location}")
            return True
        
        # Generate filename from URL
        parsed_url = urlparse(pdf_url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or not filename.lower().endswith('.pdf'):
            filename = f"document_{hash(pdf_url) % 10000}.pdf"
        
        filename = sanitize_filename(filename)
        filepath = os.path.join(download_dir, filename)
        
        # Check if file exists locally (in case tracking file was lost)
        if os.path.exists(filepath):
            logging.info(f"○ File exists locally: {filename}")
            tracker.mark_downloaded(pdf_url, filepath, source_url)
            return True
        
        if DRY_RUN:
            logging.info(f"[DRY RUN] Would download: {pdf_url} -> {filepath}")
            return True
        
        logging.info(f"↓ Downloading: {pdf_url}")
        
        # Download the PDF using fallback strategies
        response = download_with_fallback(session, pdf_url)
        if not response:
            logging.error(f"✗ All download strategies failed for {pdf_url}")
            return False
        
        # Verify it's actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type:
            first_bytes = next(iter(response.iter_content(chunk_size=10)), b'')
            if not first_bytes.startswith(b'%PDF'):
                logging.warning(f"⚠ File doesn't appear to be a PDF, skipping: {pdf_url}")
                return False
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        logging.info(f"✓ Downloaded {filename} ({file_size:,} bytes)")
        
        # Mark as downloaded in tracker
        tracker.mark_downloaded(pdf_url, filepath, source_url)
        
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"✗ Error downloading {pdf_url}: {e}")
        return False
    except Exception as e:
        logging.error(f"✗ Unexpected error downloading {pdf_url}: {e}")
        return False

def load_urls(urls_file):
    """Load URLs from the JSON file, keeping category structure."""
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls_data = json.load(f)
        return urls_data
    except FileNotFoundError:
        logging.error(f"URLs file {urls_file} not found")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file: {e}")
        return {}

def main():
    """Main function to orchestrate the PDF downloading process."""
    # Create base download directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Initialize download tracker
    tracker = DownloadTracker(TRACKING_FILE)
    
    # Load URLs with categories
    urls_data = load_urls(URLS_FILE)
    if not urls_data:
        logging.error("No URLs loaded. Exiting.")
        return
    
    # Count total URLs
    total_urls = sum(len(urls) for urls in urls_data.values())
    
    if DRY_RUN:
        logging.info("=" * 60)
        logging.info("DRY RUN MODE - No files will be downloaded")
        logging.info("=" * 60)
    
    logging.info(f"Loaded {total_urls} URLs across {len(urls_data)} categories")
    logging.info(f"Previously downloaded: {len(tracker.downloaded_pdfs)} PDFs")
    
    # Create session
    session = create_session()
    
    total_pdfs_found = 0
    total_pdfs_downloaded = 0
    total_pdfs_skipped = 0
    pages_with_no_pdfs = 0
    url_counter = 0
    
    try:
        for category, urls in urls_data.items():
            # Create category folder (sanitized)
            category_folder = sanitize_filename(category.lower().replace(' ', '_'))
            category_path = os.path.join(DOWNLOAD_DIR, category_folder)
            os.makedirs(category_path, exist_ok=True)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Category: {category} ({len(urls)} URLs)")
            logging.info(f"{'='*60}")
            
            for url in urls:
                url_counter += 1
                logging.info(f"\n[{url_counter}/{total_urls}] Processing: {url}")
                
                # Create URL-specific folder
                url_folder = create_url_folder_name(url)
                url_path = os.path.join(category_path, url_folder)
                os.makedirs(url_path, exist_ok=True)
                
                # Find PDF links on this page (main content only)
                pdf_links = find_pdf_links(session, url)
                
                if not pdf_links:
                    pages_with_no_pdfs += 1
                    logging.info("  → No PDFs to download from this page")
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    continue
                
                total_pdfs_found += len(pdf_links)
                
                # Download each PDF to the URL-specific folder
                for pdf_url in pdf_links:
                    if tracker.is_downloaded(pdf_url):
                        total_pdfs_skipped += 1
                    elif download_pdf(session, pdf_url, url_path, tracker, url):
                        total_pdfs_downloaded += 1
                    
                    # Add delay between downloads
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                
                # Add delay between pages
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
    except KeyboardInterrupt:
        logging.info("\n⚠ Download interrupted by user")
    except Exception as e:
        logging.error(f"✗ Unexpected error in main loop: {e}")
    finally:
        session.close()
        
        # Print summary
        logging.info(f"\n{'='*60}")
        logging.info(f"=== Download Summary ===")
        logging.info(f"{'='*60}")
        logging.info(f"URLs processed: {url_counter}/{total_urls}")
        logging.info(f"Pages with no PDFs in main content: {pages_with_no_pdfs}")
        logging.info(f"Total PDF links found in main content: {total_pdfs_found}")
        logging.info(f"PDFs downloaded this session: {total_pdfs_downloaded}")
        logging.info(f"PDFs skipped (already downloaded): {total_pdfs_skipped}")
        logging.info(f"Total unique PDFs tracked: {len(tracker.downloaded_pdfs)}")
        logging.info(f"Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
        logging.info(f"Tracking file: {os.path.abspath(TRACKING_FILE)}")
        logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()