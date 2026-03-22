"""
Orchestrates crawling + scraping + writing Reports (date-only version).
"""

import sys
import traceback
from datetime import date
from typing import List, Dict, Any, Set
import logging
import asyncio

# Add the project root to the path if needed
sys.path.append('.')

from data_pipeline.database.models import Report
from data_pipeline.scraping.crawler import crawl
from data_pipeline.scraping.scraper import scrape
from data_pipeline.database.sheets_interface import (
    add_reports, 
    report_url_exists,
    get_sheet,
    REPORTS_SHEET_NAME  # ADD THIS
)
from utilities.date_utils import parse_date_string
from utilities.text_utils import extract_title, extract_location, extract_actors, clean_article_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def crawl_wafa_async(max_pages: int = 5, delay: float = 1.0) -> List[Dict[str, Any]]:
    """
    Crawl WAFA website using your existing crawler.
    """
    logger.info(f"Crawling WAFA (max pages: {max_pages}, delay: {delay}s)")
    
    start_url = "https://english.wafa.ps/Pages/LastNews"
    
    try:
        # Use your existing async crawler
        article_urls = await crawl(start_url, max_pages=max_pages, delay=delay)
        logger.info(f"[STAGE 1] Found {len(article_urls)} WAFA article URLs")

        # Scrape the articles
        articles = await scrape(article_urls)
        logger.info(f"[STAGE 2] Successfully scraped {len(articles)} WAFA articles")

        # Log a sample article so we can verify content shape
        if articles:
            sample = articles[0]
            logger.info(
                "[STAGE 2 SAMPLE] url=%s | title=%s | date=%s | text_len=%d | keys=%s",
                sample.get("url", ""),
                sample.get("title", "")[:80],
                sample.get("date", ""),
                len(sample.get("text", "") or ""),
                list(sample.keys()),
            )

        return articles
    except Exception as e:
        logger.error(f"Failed to crawl/scrape WAFA: {e}\n{traceback.format_exc()}")
        return []


async def crawl_btselem_async(max_pages: int = 5, delay: float = 1.0) -> List[Dict[str, Any]]:
    """
    Crawl B'Tselem website.
    TODO: Update with actual B'Tselem start URL and scraper logic
    """
    logger.info(f"Crawling B'Tselem (max pages: {max_pages}, delay: {delay}s)")
    
    # Placeholder - need to implement B'Tselem specific crawling
    # For now, return empty list
    logger.warning("B'Tselem crawling not yet implemented")
    return []


async def crawl_ocha_async(max_pages: int = 5, delay: float = 1.0) -> List[Dict[str, Any]]:
    """
    Crawl OCHA website.
    TODO: Update with actual OCHA start URL and scraper logic
    """
    logger.info(f"Crawling OCHA (max pages: {max_pages}, delay: {delay}s)")
    
    # Placeholder - need to implement OCHA specific crawling
    # For now, return empty list
    logger.warning("OCHA crawling not yet implemented")
    return []


def article_to_report(article: Dict[str, Any], source_name: str) -> Report:
    """
    Convert raw article data into a standardized Report object (date-only version).
    
    Args:
        article: Dictionary with keys like 'url', 'date', 'text', etc.
        source_name: Name of the source (WAFA, B'Tselem, OCHA)
    
    Returns:
        Report object with standardized fields
    """
    try:
        # Parse date (date-only)
        date_str = article.get('date')
        published_date = parse_date_string(date_str)
        
        if not published_date:
            # Fallback: use current date with warning
            logger.warning(f"No valid date found for {article.get('url')}, using current date")
            published_date = date.today()
        
        # Clean and extract text
        raw_text = article.get('text', '')
        cleaned_text = clean_article_text(raw_text)
        
        # Extract title from cleaned text if not already in article
        title = article.get('title', '')
        if not title and cleaned_text:
            title = extract_title(cleaned_text) or "Untitled Article"
        
        # Extract location and actors from cleaned text
        location_raw = extract_location(cleaned_text) or ""
        actors_raw = extract_actors(cleaned_text) or ""
        
        return Report(
            source_name=source_name.upper(),
            url=article.get('url', ''),
            title=title,
            body=cleaned_text,
            published_date=published_date,  # Date only
            language=article.get('language', 'en'),
            location_raw=location_raw,
            actors_raw=actors_raw
        )
    except Exception as e:
        logger.error(f"Failed to convert article to Report: {e}")
        raise


def get_all_existing_urls() -> Set[str]:
    """Get all URLs from Reports sheet once."""
    ws = get_sheet(REPORTS_SHEET_NAME)
    # Assuming URL is column D (4th column)
    all_urls = ws.col_values(4)[1:]  # Skip header row

    # Filter out None/empty values and ensure they're strings
    url_set = set()
    for url in all_urls:
        if url is not None:
            url_str = str(url).strip()
            if url_str:  # Only add non-empty strings
                url_set.add(url_str)

    return url_set


def filter_existing_reports(articles: List[Dict[str, Any]], source_name: str) -> List[Report]:
    """Convert articles to Reports and filter out duplicates."""
    logger.info("[STAGE 3] filter_existing_reports: received %d articles", len(articles))

    # --- Stage 3a: fetch existing URLs from Google Sheets ---
    try:
        existing_urls = get_all_existing_urls()
        logger.info("[STAGE 3a] Fetched %d existing URLs from Sheets", len(existing_urls))
    except Exception as e:
        logger.error(
            "[STAGE 3a] FAILED to fetch existing URLs from Sheets — "
            "cannot deduplicate. Full error:\n%s",
            traceback.format_exc(),
        )
        raise  # re-raise so the caller sees it

    reports = []
    new_count = 0
    duplicate_count = 0
    error_count = 0

    for i, article in enumerate(articles):
        url = article.get('url')
        if not url:
            logger.warning("[STAGE 3b] Article %d has no URL — skipping. Keys: %s", i, list(article.keys()))
            error_count += 1
            continue

        url_str = str(url).strip()
        if not url_str:
            logger.warning("[STAGE 3b] Article %d has empty URL string — skipping.", i)
            error_count += 1
            continue

        # Check for duplicates
        if url_str in existing_urls:
            logger.debug("[STAGE 3b] DUPLICATE: %s", url_str)
            duplicate_count += 1
            continue

        # --- Stage 3c: convert to Report object ---
        try:
            report = article_to_report(article, source_name)
            reports.append(report)
            new_count += 1
            existing_urls.add(url_str)
        except Exception as e:
            logger.error(
                "[STAGE 3c] article_to_report FAILED for %s:\n%s",
                url_str,
                traceback.format_exc(),
            )
            error_count += 1

    logger.info(
        "[STAGE 3 DONE] %d articles → %d new, %d duplicates, %d errors",
        len(articles), new_count, duplicate_count, error_count,
    )
    return reports


async def ingest_source_async(source_name: str, max_pages: int = 5, delay: float = 1.0) -> int:
    """
    Ingest reports from a single source asynchronously (date-only version).
    
    Args:
        source_name: One of 'wafa', 'btselem', 'ocha'
        max_pages: Maximum number of pages to crawl
        delay: Delay between requests in seconds
    
    Returns:
        Number of new reports added
    """
    logger.info(f"Starting async ingest for {source_name} (max pages: {max_pages})")
    
    # Crawl the source
    if source_name.lower() == 'wafa':
        articles = await crawl_wafa_async(max_pages, delay)
    elif source_name.lower() == 'btselem':
        articles = await crawl_btselem_async(max_pages, delay)
    elif source_name.lower() == 'ocha':
        articles = await crawl_ocha_async(max_pages, delay)
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    if not articles:
        logger.warning(f"No articles found for {source_name}")
        return 0

    # Convert to Reports and filter duplicates
    try:
        reports = filter_existing_reports(articles, source_name.upper())
    except Exception as e:
        logger.error(
            "[STAGE 3] filter_existing_reports raised an exception — 0 reports saved.\n%s",
            traceback.format_exc(),
        )
        return 0

    if not reports:
        logger.info(f"No new reports to add for {source_name}")
        return 0

    # Add to Google Sheets
    logger.info("[STAGE 4] Writing %d new reports to Google Sheets ...", len(reports))
    try:
        add_reports(reports)
        logger.info(f"[STAGE 4 DONE] Successfully added {len(reports)} new reports from {source_name}")
        return len(reports)
    except Exception as e:
        logger.error(f"[STAGE 4] FAILED to write to Google Sheets:\n{traceback.format_exc()}")
        return 0


async def ingest_all_sources_async(max_pages: int = 5, delay: float = 1.0) -> Dict[str, int]:
    """
    Ingest reports from all sources asynchronously.
    
    Args:
        max_pages: Maximum number of pages to crawl per source
        delay: Delay between requests in seconds
    
    Returns:
        Dictionary with counts of new reports per source
    """
    logger.info(f"Starting async ingest for all sources (max pages: {max_pages}, delay: {delay}s)")
    
    results = {}
    sources = ['wafa']  # Start with WAFA only for now
    
    for source in sources:
        try:
            count = await ingest_source_async(source, max_pages, delay)
            results[source] = count
        except Exception as e:
            logger.error(f"Failed to ingest {source}: {e}")
            results[source] = 0
    
    total_new = sum(results.values())
    logger.info(f"Async ingest complete. Total new reports: {total_new}")
    logger.info(f"Breakdown: {results}")
    
    return results


def ingest_source(source_name: str, max_pages: int = 5, delay: float = 1.0) -> int:
    """
    Synchronous wrapper for ingest_source_async.
    """
    return asyncio.run(ingest_source_async(source_name, max_pages, delay))


def ingest_all_sources(max_pages: int = 5, delay: float = 1.0) -> Dict[str, int]:
    """
    Synchronous wrapper for ingest_all_sources_async.
    """
    return asyncio.run(ingest_all_sources_async(max_pages, delay))


if __name__ == "__main__":
    """
    Main entry point for running the ingest pipeline.
    
    Example usage:
        python -m data_pipeline.scraping.ingest_reports
    """
    
    # Run the async pipeline
    print("Starting report ingestion pipeline...")
    print("=" * 60)
    
    results = ingest_all_sources(max_pages=2, delay=0.5)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    
    print(f"\nResults: {results}")
    print(f"Total new reports added: {sum(results.values())}")
    
    if sum(results.values()) == 0:
        print("\nNote: No new reports were added.")
        print("This could mean:")
        print("1. No new articles were found")
        print("2. All found articles were duplicates")
        print("3. There was an error during scraping")