import requests
from bs4 import BeautifulSoup
import io
import os
import tempfile
from typing import Dict, Any, Tuple, Optional, List, Set
import logging
from urllib.parse import urlparse, urljoin
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_web_content(url: str, follow_links: bool = False, max_links: int = 5) -> Tuple[str, Dict[Any, Any], str]:
    """
    Fetch content from a URL and determine its type.
    
    Args:
        url (str): The URL to fetch content from
        follow_links (bool): Whether to follow links within the page
        max_links (int): Maximum number of links to follow
        
    Returns:
        Tuple[str, Dict[Any, Any], str]: Tuple containing:
            - content (str): The extracted text content
            - metadata (Dict): Metadata about the content
            - content_type (str): The type of content (html, pdf)
    """
    try:
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Extract domain for metadata
        domain = urlparse(url).netloc
        
        # Get content type from headers
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Check if it's a PDF
        if 'application/pdf' in content_type:
            content, metadata = process_online_pdf(response.content, url, domain)
            return content, metadata, 'pdf'
        
        # Default to HTML processing
        else:
            content, metadata, links = process_html_content(response.text, url, domain)
            
            # Follow links if requested
            if follow_links and links:
                linked_content = follow_page_links(links, domain, max_links)
                if linked_content:
                    # Append linked content to the main content
                    content += "\n\n--- LINKED CONTENT ---\n\n" + linked_content
                    metadata["includes_linked_content"] = True
                    metadata["linked_pages_count"] = len(linked_content)
            
            return content, metadata, 'html'
            
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise ValueError(f"Failed to fetch content from {url}: {str(e)}")

def process_html_content(html_content: str, url: str, domain: str) -> Tuple[str, Dict[Any, Any], List[str]]:
    """
    Process HTML content to extract clean text and metadata.
    
    Args:
        html_content (str): The HTML content to process
        url (str): The source URL
        domain (str): The domain of the URL
        
    Returns:
        Tuple[str, Dict[Any, Any], List[str]]: The extracted text, metadata, and list of links
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "Untitled"
        
        # Extract links before removing elements
        links = extract_important_links(soup, url, domain)
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
            
        # Get text and clean it up
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Create metadata
        metadata = {
            "source": url,
            "domain": domain,
            "title": title,
            "content_type": "html",
            "source_type": "web"
        }
        
        return text, metadata, links
        
    except Exception as e:
        logger.error(f"Error processing HTML from {url}: {str(e)}")
        raise ValueError(f"Failed to process HTML content from {url}: {str(e)}")

def extract_important_links(soup: BeautifulSoup, base_url: str, domain: str) -> List[str]:
    """
    Extract important links from a webpage.
    
    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page
        base_url (str): The base URL of the page
        domain (str): The domain of the URL
        
    Returns:
        List[str]: List of important links
    """
    important_links = []
    
    # Find all links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href', '')
        
        # Skip empty links, anchors, javascript, and mailto links
        if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
            continue
        
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        
        # Parse the URL
        parsed_url = urlparse(full_url)
        
        # Only include links from the same domain
        if parsed_url.netloc == domain:
            # Exclude common navigation, login, and utility pages
            path = parsed_url.path.lower()
            if any(excluded in path for excluded in ['/login', '/signup', '/register', '/contact', '/about', '/terms', '/privacy']):
                continue
                
            # Prioritize links that appear to be content pages
            # Look for links with text that suggests they're important content
            link_text = a_tag.get_text().strip().lower()
            
            # Check if the link is likely to be important content
            if len(link_text) > 5 and not any(common in link_text.lower() for common in ['sign in', 'log in', 'register', 'contact', 'about us']):
                important_links.append(full_url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in important_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    return unique_links

def follow_page_links(links: List[str], domain: str, max_links: int = 5) -> str:
    """
    Follow links and extract content from linked pages.
    
    Args:
        links (List[str]): List of links to follow
        domain (str): The domain of the original URL
        max_links (int): Maximum number of links to follow
        
    Returns:
        str: Combined content from linked pages
    """
    if not links:
        return ""
    
    # Limit the number of links to follow
    links_to_follow = links[:max_links]
    
    all_linked_content = []
    
    for i, link in enumerate(links_to_follow):
        try:
            logger.info(f"Following link {i+1}/{len(links_to_follow)}: {link}")
            
            # Add a small delay to avoid overwhelming the server
            if i > 0:
                time.sleep(1)
            
            # Fetch content from the linked page
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            
            response = requests.get(link, headers=headers, timeout=20)
            response.raise_for_status()
            
            # Only process HTML content
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                continue
            
            # Process the linked page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "Untitled"
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
                script_or_style.decompose()
                
            # Get text and clean it up
            text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Add the linked content with a title
            linked_page_content = f"--- Content from {title} ({link}) ---\n\n{clean_text}\n\n"
            all_linked_content.append(linked_page_content)
            
        except Exception as e:
            logger.error(f"Error following link {link}: {str(e)}")
            continue
    
    return "\n".join(all_linked_content)

def process_online_pdf(pdf_content: bytes, url: str, domain: str) -> Tuple[str, Dict[Any, Any]]:
    """
    Process PDF content from a URL.
    
    Args:
        pdf_content (bytes): The PDF content as bytes
        url (str): The source URL
        domain (str): The domain of the URL
        
    Returns:
        Tuple[str, Dict[Any, Any]]: The extracted text and metadata
    """
    try:
        # Import PyPDF2 here to avoid dependency issues if not installed
        import PyPDF2
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name
        
        # Extract text from PDF
        text = ""
        pdf_title = os.path.basename(url)
        
        try:
            with open(temp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Try to get PDF title from metadata
                if reader.metadata and reader.metadata.title:
                    pdf_title = reader.metadata.title
                
                # Extract text from all pages
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Create metadata
        metadata = {
            "source": url,
            "domain": domain,
            "title": pdf_title,
            "content_type": "pdf",
            "source_type": "web"
        }
        
        return text, metadata
        
    except ImportError:
        logger.error("PyPDF2 is not installed. Cannot process PDF files.")
        raise ValueError("PyPDF2 is required for processing PDF files but is not installed.")
    except Exception as e:
        logger.error(f"Error processing PDF from {url}: {str(e)}")
        raise ValueError(f"Failed to process PDF content from {url}: {str(e)}")

async def process_batch_urls(urls: list, process_func) -> list:
    """
    Process a batch of URLs asynchronously.
    
    Args:
        urls (list): List of URLs to process
        process_func: Function to process each URL
        
    Returns:
        list: List of results
    """
    results = []
    for url in urls:
        try:
            result = await process_func(url)
            results.append({"url": url, "status": "success", "result": result})
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})
    
    return results
