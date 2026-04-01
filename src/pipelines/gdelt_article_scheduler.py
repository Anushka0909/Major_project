"""
Automated GDELT article fetcher that bypasses BigQuery quota.
Runs as a background service, fetching articles periodically.

Usage:
    # Run once
    python src/pipelines/gdelt_article_scheduler.py --once
    
    # Run as daemon (keeps running, fetches every hour)
    python src/pipelines/gdelt_article_scheduler.py --daemon
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import json
from typing import List, Dict
import schedule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.helpers import ensure_directory

logger = get_logger(__name__)
settings = get_settings()


class GDELTArticleFetcher:
    """
    Fetch GDELT article metadata without BigQuery.
    Uses GDELT's public API and news aggregation services.
    """
    
    def __init__(self):
        self.output_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment"
        ensure_directory(self.output_dir)
        self.articles_file = self.output_dir / "articles.csv"
        
        # GDELT DOC API endpoint (free, no quota!)
        self.gdelt_doc_api = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def fetch_articles_for_country_pair(
        self,
        country1: str,
        country2: str,
        max_articles: int = 10
    ) -> List[Dict]:
        """
        Fetch recent news articles mentioning both countries.
        Uses GDELT DOC 2.0 API with retry logic.
        """
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Build query - use country names instead of codes for better results
                country_names = {
                    'IND': 'India', 'USA': 'United States', 'CHN': 'China',
                    'ARE': 'UAE', 'DEU': 'Germany', 'GBR': 'United Kingdom',
                    'JPN': 'Japan', 'SGP': 'Singapore', 'HKG': 'Hong Kong'
                }
                
                c1_name = country_names.get(country1, country1)
                c2_name = country_names.get(country2, country2)
                
                query = f'"{c1_name}" AND "{c2_name}" (trade OR export OR import)'
                
                params = {
                    'query': query,
                    'mode': 'artlist',
                    'maxrecords': max_articles,
                    'format': 'json',
                    'timespan': '7d'
                }
                
                response = requests.get(
                    self.gdelt_doc_api,
                    params=params,
                    timeout=30,
                    headers={'User-Agent': 'TradeForecasting/1.0'}
                )
                
                # Check for rate limit
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"  Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"  Max retries reached for {country1}-{country2}")
                        return []
                
                response.raise_for_status()
                data = response.json()
                
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'date': article.get('seendate', datetime.now().strftime('%Y%m%d')),
                        'domain': article.get('domain', ''),
                        'language': article.get('language', 'en'),
                        'country_1_iso3': country1,
                        'country_2_iso3': country2,
                        'sentiment': float(article.get('tone', 0.0)),
                        'fetched_at': datetime.now().isoformat()
                    })
                
                return articles
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"  Error, retrying: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"  Failed after {max_retries} attempts: {e}")
                    return []
        
        return []
    
    def get_priority_country_pairs(self) -> List[tuple]:
        """
        Get list of country pairs to fetch articles for.
        Prioritizes India's trade partners.
        """
        # Priority: India's major trade partners
        india_partners = [
            'USA', 'CHN', 'ARE', 'HKG', 'SAU', 'SGP',
            'DEU', 'GBR', 'NLD', 'BEL', 'FRA', 'ITA',
            'JPN', 'KOR', 'MYS', 'IDN', 'THA', 'VNM'
        ]
        
        pairs = []
        
        # India to each partner
        for partner in india_partners:
            pairs.append(('IND', partner))
        
        # Major global trade pairs (for context)
        major_pairs = [
            ('USA', 'CHN'), ('USA', 'MEX'), ('DEU', 'CHN'),
            ('CHN', 'JPN'), ('USA', 'CAN'), ('DEU', 'FRA')
        ]
        
        pairs.extend(major_pairs)
        
        return pairs
    
    def fetch_all_articles(self, max_per_pair: int = 10) -> pd.DataFrame:
        """
        Fetch articles for all priority country pairs.
        
        Args:
            max_per_pair: Max articles per country pair
        
        Returns:
            DataFrame with all fetched articles
        """
        logger.info("Fetching articles for priority country pairs...")
        
        pairs = self.get_priority_country_pairs()
        all_articles = []
        
        for i, (c1, c2) in enumerate(pairs, 1):
            logger.info(f"  [{i}/{len(pairs)}] Fetching {c1}-{c2}...")
            
            articles = self.fetch_articles_for_country_pair(c1, c2, max_per_pair)
            all_articles.extend(articles)
            
            # Be nice to API
            time.sleep(2)
        
        if not all_articles:
            logger.warning("No articles fetched!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_articles)
        logger.info(f"Fetched {len(df):,} total articles")
        
        return df
    
    def save_articles(self, df: pd.DataFrame):
        """
        Save articles to CSV, appending to existing if present.
        """
        if len(df) == 0:
            return
        
        # Remove duplicates by URL
        df = df.drop_duplicates(subset=['url'], keep='last')
        
        # If file exists, merge with existing
        if self.articles_file.exists():
            try:
                existing = pd.read_csv(self.articles_file)
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['url'], keep='last')
                combined = combined.sort_values('date', ascending=False)
                
                # Keep only last 1000 articles to avoid huge file
                combined = combined.head(1000)
                
                combined.to_csv(self.articles_file, index=False)
                logger.info(f"Updated {self.articles_file} ({len(combined):,} total articles)")
            except Exception as e:
                logger.error(f"Failed to merge with existing: {e}")
                df.to_csv(self.articles_file, index=False)
        else:
            df.to_csv(self.articles_file, index=False)
            logger.info(f"Created {self.articles_file} ({len(df):,} articles)")
    
    def run_once(self):
        """Run article fetching once."""
        logger.info("="*60)
        logger.info("GDELT ARTICLE FETCHER - Single Run")
        logger.info("="*60)
        
        df = self.fetch_all_articles(max_per_pair=10)
        self.save_articles(df)
        
        logger.info("="*60)
        logger.info("✅ Article fetch complete!")
        logger.info("="*60)


def run_scheduled_fetch():
    """Run the scheduled fetch."""
    fetcher = GDELTArticleFetcher()
    
    try:
        fetcher.run_once()
    except Exception as e:
        logger.error(f"Scheduled fetch failed: {e}", exc_info=True)


def run_daemon():
    """
    Run as daemon - fetches articles every hour.
    """
    logger.info("="*60)
    logger.info("GDELT ARTICLE FETCHER - Daemon Mode")
    logger.info("="*60)
    logger.info("Will fetch articles every hour")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60)
    
    # Schedule to run every hour
    schedule.every().hour.do(run_scheduled_fetch)
    
    # Run immediately on start
    run_scheduled_fetch()
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("\n👋 Daemon stopped by user")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch GDELT articles without BigQuery quota limits'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon (fetch every hour)'
    )
    parser.add_argument(
        '--max-per-pair',
        type=int,
        default=10,
        help='Max articles per country pair (default: 10)'
    )
    
    args = parser.parse_args()
    
    if args.daemon:
        run_daemon()
    else:
        # Default: run once
        fetcher = GDELTArticleFetcher()
        df = fetcher.fetch_all_articles(max_per_pair=args.max_per_pair)
        fetcher.save_articles(df)
        
        print("\n" + "="*60)
        print("✅ FETCH COMPLETE")
        print("="*60)
        print(f"Articles saved: {len(df):,}")
        print(f"Output: {fetcher.articles_file}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()