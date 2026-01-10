import asyncio
import time
from bs4 import BeautifulSoup
import re
import requests
from playwright.async_api import async_playwright
from typing import Dict, Any, Optional


def get_date(text: str) -> Optional[str]:
    """
    Extract date from text using regex.
    
    Args:
        text: Text to search for dates
    
    Returns:
        Date string or None if not found
    """
    if not text:
        return None
    
    date_regex = re.compile(
        r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
        r'January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:,?\s+\d{4})?'
        r'|\b\d{4}-\d{2}-\d{2}\b'
    )
    match = date_regex.search(text)
    return match.group(0) if match else None


def extract_wafa(url: str) -> Optional[Dict[str, Any]]:
    """
    Extract article data from WAFA website.
    Only extracts West Bank articles.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check if relevant (West Bank region ONLY)
        TARGET_HREF = "/Regions/Details/2"
        
        # Look for the region/category meta info
        meta_block = soup.select_one("div.single-blog.mb-50 > div.blog-wrap > div.meta")
        if not meta_block:
            return None 
        
        # Check ALL category links to ensure it's ONLY West Bank
        category_links = meta_block.select("a.meta-item.category")
        is_west_bank = False
        has_other_region = False
        
        for a in category_links:
            href = a.get("href", "")
            if href and isinstance(href, str):
                if href.startswith(TARGET_HREF):
                    is_west_bank = True
                elif href.startswith("/Regions/Details/"):
                    # Check if it's another region (Gaza, Jerusalem, etc.)
                    if not href.startswith("/Regions/Details/2"):
                        has_other_region = True
                        break  # Exit early if we find a non-West Bank region
        
        # Only include if it's West Bank AND doesn't have other regions
        if not is_west_bank or has_other_region:
            return None

        # Extract title
        title_element = soup.select_one("h1.title")
        title = title_element.get_text(strip=True) if title_element else ""

        # Extract text from paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Filter very short paragraphs
                paragraphs.append(text)
        
        text = "\n\n".join(paragraphs)
        
        # Extract date
        date = None
        meta_text = meta_block.get_text()
        date = get_date(meta_text)
        
        if not date:
            date_text = soup.get_text()
            date = get_date(date_text)
        
        if not date:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "url": url,
            "date": date,
            "title": title,
            "text": text,
            "language": "en",
            "source": "WAFA"
        }
    except Exception as e:
        print(f"Error scraping WAFA article {url}: {e}")
        return None


async def goto(page, url, retries=3):
    """
    Navigate to URL with retry logic.
    """
    for attempt in range(retries):
        try:
            return await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"[Goto error] Attempt {attempt+1}/{retries} for {url}: {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2)


async def extract_playwright(url: str) -> Optional[Dict[str, Any]]:
    """
    Extract article data using Playwright for JavaScript-heavy sites.
    
    Args:
        url: Article URL
    
    Returns:
        Dictionary with url, date, text, title, and language
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await goto(page, url)

            # Extract title
            title = await page.evaluate("""
                () => {
                    const h1 = document.querySelector('h1');
                    return h1 ? h1.innerText.trim() : document.title;
                }
            """)

            # Extract paragraphs
            paragraphs = await page.evaluate("""
                () => {
                    const paras = [];
                    document.querySelectorAll("p").forEach(p => {
                        const text = p.innerText.trim();
                        if (text.length > 40) {
                            paras.push(text);
                        }
                    });
                    return paras;
                }
            """)
            text = "\n\n".join(paragraphs)
            
            # Extract date from page text
            date_text = await page.evaluate("() => document.body.innerText")
            date = get_date(date_text)
            
            # Fallback date if not found
            if not date:
                from datetime import datetime
                date = datetime.now().strftime("%Y-%m-%d")
            
            await browser.close()
            
            return {
                "url": url,
                "date": date,
                "title": title,
                "text": text,
                "language": "en",
                "source": "Unknown"  # Should be determined by caller
            }
    except Exception as e:
        print(f"Error scraping with Playwright {url}: {e}")
        return None


async def scrape(article_urls: list) -> list:
    """
    Scrape multiple articles.
    
    Args:
        article_urls: List of article URLs
    
    Returns:
        List of article dictionaries
    """
    articles = []
    
    for url in article_urls:
        if "wafa" in url.lower():
            article = extract_wafa(url)
        else:
            article = await extract_playwright(url)
        
        if article:
            articles.append(article)
            print(f"✓ Scraped: {url}")
        else:
            print(f"✗ Failed: {url}")
    
    return articles