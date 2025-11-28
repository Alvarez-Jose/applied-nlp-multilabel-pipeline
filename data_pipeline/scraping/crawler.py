import asyncio
import aiohttp
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
from collections import deque


def is_article_url(url):
    # WAFA article URLs typically include /Pages/Details/######
    url = url.lower()
    return any(["pages/details/" in url])


async def fetch(session, url, semaphore, headers):
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

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AsyncPoliteCrawler/1.0)"}
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
                norm = urlparse(link)._replace(fragment="").geturl()

                if norm not in visited:
                    queue.append(norm)
                
                # checking if url is valid and save articles
                if is_article_url(norm):
                    found_articles.add(norm)

    return list(found_articles)