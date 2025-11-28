import asyncio
import aiohttp
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
from collections import deque
import re
import requests
from playwright.async_api import async_playwright


# Scraper
def get_date(text):
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


def extract_wafa(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)
    date_text = soup.get_text()
    date = get_date(date_text)
    return {"date": date, "text": text}


async def goto(page, url, retries=3):
    for attempt in range(retries):
        try:
            return await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"[Goto error] Attempt {attempt+1}/{retries} for {url}: {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2)


async def extract_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await goto(page, url)

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

        date_text = await page.evaluate("() => document.body.innerText")
        date = get_date(date_text)

        await browser.close()
        return {"date": date, "text": "\n\n".join(paragraphs)}


# Crawler
def is_article_url(url):
    """
    WAFA article URLs typically include /Details/######
    """
    url = url.lower()
    return any([
        "pages/details/" in url,
        "/article/" in url,
        "/news/" in url,
    ])


TARGET_HREF = "/Regions/Details/2"
async def article_is_relevant(session, url):
    try:
        async with session.get(url, timeout=15) as resp:
            html = await resp.text()
    except:
        return False

    soup = BeautifulSoup(html, "html.parser")

    meta_block = soup.select_one("div.single-blog.mb-50 > div.blog-wrap > div.meta")
    if not meta_block:
        return False

    for a in meta_block.select("a.meta-item.category"):
        href = a.get("href", "")
        if href.startswith(TARGET_HREF):
            return True
    return False


async def fetch(session, url, semaphore, headers):
    """Download a page politely (with semaphore rate limiting)."""
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    return None
                return await resp.text()
        except:
            return None


async def crawl(start_url, max_pages=300, delay=1.0):
    parsed = urlparse(start_url)
    domain = parsed.scheme + "://" + parsed.netloc

    rp = robotparser.RobotFileParser()
    robots_url = domain + "/robots.txt"

    try:
        rp.set_url(robots_url)
        rp.read()
        print(f"Loaded robots.txt from {robots_url}")
    except:
        print("Failed loading robots.txt — assuming allowed.")

    visited = set()
    queue = deque([start_url])
    found_articles = set()

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AsyncPoliteCrawler/1.0)"
    }

    semaphore = asyncio.Semaphore(5)  

    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < max_pages:
            url = queue.popleft()

            if not rp.can_fetch(headers["User-Agent"], url):
                print("Blocked by robots.txt:", url)
                continue

            if url in visited:
                continue

            visited.add(url)
            print(f"[Crawl] {len(visited)}/{max_pages} → {url}")

            html = await fetch(session, url, semaphore, headers)
            await asyncio.sleep(delay)

            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                parsed_link = urlparse(link)
                norm = parsed_link._replace(fragment="").geturl()

                if parsed_link.netloc != parsed.netloc:
                    continue

                if norm not in visited:
                    queue.append(norm)
                
                # checking if url is valid and article is relevant, save articles
                if is_article_url(norm) and await article_is_relevant(session, norm):
                    found_articles.add(norm)

    return list(found_articles)


async def main():
    START_URL = "https://english.wafa.ps/Regions/Details/2?pageNumber=0"

    print("Crawling")
    article_urls = await crawl(START_URL, max_pages=1, delay=1.0)

    print(f"\nFound {len(article_urls)} article URLs.\n")

    print("Scraping")
    for idx, url in enumerate(article_urls[:20]):
        print(f"Article {idx+1}")
        print("URL:", url)

        if "wafa.ps" in url:
            article = extract_wafa(url)
        else:
            article = await extract_playwright(url)

        print("DATE:", article["date"])
        print("TEXT:", article["text"][:100], "...")

if __name__ == "__main__":
    asyncio.run(main())
