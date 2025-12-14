import asyncio
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urljoin
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Add config path
sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

def extract_pdf_links(content: str, source_url: str) -> list[str]:
    """
    Extract PDF links from markdown content using regex.
    
    Args:
        content: Markdown content to scan for PDF links
        source_url: Source URL for resolving relative links
        
    Returns:
        List of full PDF URLs found in the content
    """
    pdf_links = []
    
    # Regex patterns to find PDF links in markdown
    # Pattern 1: [text](url.pdf) - markdown links
    markdown_pattern = r'\[([^\]]*)\]\(([^)]*\.pdf[^)]*)\)'
    
    # Pattern 2: <a href="url.pdf"> - HTML links
    html_pattern = r'<a[^>]+href=["\']([^"\']*\.pdf[^"\']*)["\'][^>]*>'
    
    # Pattern 3: Direct URLs ending in .pdf
    direct_pattern = r'https?://[^\s<>"]+\.pdf(?:\?[^\s<>"]*)?'
    
    # Find all matches using all patterns
    markdown_matches = re.findall(markdown_pattern, content, re.IGNORECASE)
    html_matches = re.findall(html_pattern, content, re.IGNORECASE)
    direct_matches = re.findall(direct_pattern, content, re.IGNORECASE)
    
    # Process markdown links (extract URL from tuple)
    for _, url in markdown_matches:
        pdf_links.append(url.strip())
    
    # Process HTML links
    for url in html_matches:
        pdf_links.append(url.strip())
    
    # Process direct URLs
    for url in direct_matches:
        pdf_links.append(url.strip())
    
    # Convert relative URLs to absolute URLs and handle edge cases
    full_pdf_links = []
    for link in pdf_links:
        try:
            # Skip empty or malformed links
            if not link or link.isspace():
                continue
                
            # Remove any surrounding quotes or brackets
            link = link.strip('\'"()[]{}')
            
            # Skip if still empty after cleaning
            if not link:
                continue
            
            # Convert relative URLs to absolute
            if link.startswith(('http://', 'https://')):
                # Already absolute URL
                full_url = link
            elif link.startswith('//'):
                # Protocol-relative URL
                parsed_source = urlparse(source_url)
                full_url = f"{parsed_source.scheme}:{link}"
            else:
                # Relative URL - use urljoin to properly resolve
                full_url = urljoin(source_url, link)
            
            # Validate that the final URL looks reasonable
            parsed_url = urlparse(full_url)
            if parsed_url.scheme in ('http', 'https') and parsed_url.netloc:
                full_pdf_links.append(full_url)
                
        except Exception as e:
            # Handle malformed URLs gracefully - log but continue
            print(f"Warning: Could not process PDF link '{link}' from {source_url}: {e}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pdf_links = []
    for link in full_pdf_links:
        if link not in seen:
            seen.add(link)
            unique_pdf_links.append(link)
    
    return unique_pdf_links

def load_urls_from_file(filename: str) -> list[str]:
    """
    Load and parse URLs from JSON file, extracting all URLs from all categories.
    
    Args:
        filename: Path to the JSON file containing URLs
        
    Returns:
        List of all URLs from all categories
        
    Raises:
        SystemExit: If JSON file is malformed or cannot be read
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten all URLs from all categories
        all_urls = []
        for category, urls in data.items():
            if isinstance(urls, list):
                all_urls.extend(urls)
            else:
                print(f"Warning: Category '{category}' does not contain a list of URLs")
        
        print(f"Loaded {len(all_urls)} URLs from {filename}")
        return all_urls
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Malformed JSON in file '{filename}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

async def main():
    # Get config and setup paths
    config = get_config()
    project_root = Path(__file__).parent.parent.parent
    
    output_dir = project_root / "data" / "raw" / "webpages"
    urls_file = project_root / "data" / "urls.txt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load URLs from JSON file
    urls = load_urls_from_file(str(urls_file))

    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter()
        )
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=run_config
        )

        for result in results:
            if result.success:
                # Generate filename from URL
                parsed_url = urlparse(result.url)
                filename = f"{parsed_url.netloc}{parsed_url.path}".replace("/", "_").replace(".", "-")
                filepath = output_dir / f"{filename}.txt"
                
                # Write content to file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(result.markdown)
                print(f"Saved: {filepath}")
            else:
                print(f"Failed: {result.url} - {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())