#!/usr/bin/env python3
"""
Debug the ingest pipeline step by step.
"""
import sys
import asyncio
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append('.')

from data_pipeline.scraping.ingest_reports import (
    crawl_wafa_async,
    article_to_report,
    filter_existing_reports
)
from data_pipeline.database.sheets_interface import report_url_exists, add_reports

async def debug_pipeline():
    print("=" * 60)
    print("DEBUG PIPELINE")
    print("=" * 60)
    
    # Step 1: Crawl
    print("\n1. CRAWLING...")
    articles = await crawl_wafa_async(max_pages=1, delay=0.5)
    print(f"   Found {len(articles)} articles")
    
    if not articles:
        print("   No articles found!")
        return
    
    # Show first article
    print(f"\n   First article sample:")
    print(f"   URL: {articles[0].get('url')}")
    print(f"   Date: {articles[0].get('date')}")
    print(f"   Title: {articles[0].get('title', 'No title')[:50]}...")
    print(f"   Text length: {len(articles[0].get('text', ''))} chars")
    
    # Step 2: Test report_url_exists
    print("\n2. TESTING DUPLICATE CHECK...")
    test_url = articles[0].get('url')
    if test_url:
        exists = report_url_exists(test_url)
        print(f"   URL '{test_url[:50]}...' exists in sheets: {exists}")
    
    # Step 3: Convert to Report
    print("\n3. CONVERTING TO REPORT...")
    try:
        report = article_to_report(articles[0], "WAFA")
        print(f"   ✓ Successfully created Report:")
        print(f"     Source: {report.source_name}")
        print(f"     Title: {report.title[:50]}...")
        print(f"     Date: {report.published_date}")
        print(f"     Language: {report.language}")
        print(f"     Location: {report.location_raw}")
        print(f"     Actors: {report.actors_raw}")
    except Exception as e:
        print(f"   ✗ Failed to convert: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Filter existing
    print("\n4. FILTERING EXISTING REPORTS...")
    reports = filter_existing_reports(articles, "WAFA")
    print(f"   After filtering: {len(reports)} new reports")
    
    # Step 5: Test adding to sheets
    if reports:
        print("\n5. TESTING GOOGLE SHEETS CONNECTION...")
        try:
            # Test with just first report
            test_reports = reports[:1]
            print(f"   Attempting to add {len(test_reports)} report(s) to Google Sheets...")
            add_reports(test_reports)
            print(f"   ✓ Successfully added to Google Sheets!")
        except Exception as e:
            print(f"   ✗ Failed to add to Google Sheets: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(debug_pipeline())