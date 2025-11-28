import asyncio
import time
from bs4 import BeautifulSoup
import re
import requests
from playwright.async_api import async_playwright
from crawler import crawl


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

    # check if relevant
    TARGET_HREF = "/Regions/Details/2"
    meta_block = soup.select_one("div.single-blog.mb-50 > div.blog-wrap > div.meta")
    if not meta_block:
        return None 
    if not any(a.get("href", "").startswith(TARGET_HREF) for a in meta_block.select("a.meta-item.category")):
        return None

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)
    date_text = soup.get_text()
    date = get_date(date_text)
    return {"url": url, "date": date, "text": text}


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
        text = "\n\n".join(paragraphs)
        date_text = await page.evaluate("() => document.body.innerText")
        date = get_date(date_text)
        await browser.close()
        return {"url": url, "date": date, "text": text}

async def scrape(article_urls):
    articles = []
    for url in article_urls[:30]:
        if "wafa" in url:
            article = extract_wafa(url)
        else:
            article = await extract_playwright(url)
        if not article:
            continue
        articles.append(article)
    return articles