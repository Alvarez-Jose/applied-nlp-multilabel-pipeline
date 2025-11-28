from bs4 import BeautifulSoup
import requests
from playwright.async_api import async_playwright
import asyncio
import re

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
    date = match.group(0) if match else None
    return date

def extract_wafa(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)
    date_text = soup.get_text()
    date = get_date(date_text)
    return {"date": date, "text": "\n\n".join(paragraphs)}

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

        date_text = await page.evaluate("""
            () => document.body.innerText
        """)
        date = get_date(date_text)

        await browser.close()
        return {"date": date, "text": "\n\n".join(paragraphs)}


async def main():
    '''
    csv_path = "URL_CSV"
    df = pd.read_csv(csv_path)

    for col in ["Date", "Text", "Order"]:
        if col not in df.columns:
            df[col] = None
    '''


    urls = [
        "https://english.wafa.ps/Pages/Details/132595",
        "https://english.wafa.ps/Pages/Details/153747",
        "https://www.aljazeera.com/news/2023/1/5/16-year-old-palestinian-killed-israeli-forces-nablus"
    ]


    for idx, url in enumerate(urls):
        if "wafa.ps" in url.lower():
            article = extract_wafa(url)
        else:
            article = await extract_playwright(url)

        # df.at[idx+1, "Date"] = article["date"]
        # df.at[idx+1, "Text"] = article["text"]

        print(f"\n===== ARTICLE {idx+1} =====\n")
        print("DATE:", article["date"])
        print("TEXT:", article["text"][:100], "...")
        print()
    
    # df.to_csv("scraped_articles.csv")


if __name__ == "__main__":
    asyncio.run(main())