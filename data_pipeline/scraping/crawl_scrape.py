from crawler import crawl
from scraper import scrape
import asyncio

async def main():
    start_url_wafa = "https://english.wafa.ps/Regions/Details/2?pageNumber=0"
    # start_url_aj = "https://www.aljazeera.com/tag/occupied-west-bank/"

    print("Crawling")
    article_urls = await crawl(start_url_wafa, max_pages=1, delay=1.0)
    print(f"\nFound {len(article_urls)} article URLs.\n")

    articles = await scrape(article_urls)
    print(f"\nScraping {len(articles)} articles")
    for idx, article in enumerate(articles[:30]):
        print(f"Article {idx+1}")
        print("URL:", article["url"])
        print("DATE:", article["date"])
        print("TEXT:", article["text"][:100], "...")

if __name__ == "__main__":
    asyncio.run(main())