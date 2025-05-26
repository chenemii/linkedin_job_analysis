"""
EDGAR Filing Collector

Downloads and processes 10-K filings from S&P 500 companies,
with a focus on identifying M&A related content and organizational changes.
"""

import os
import re
import json
import logging
import time
import html
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

import requests
from sec_edgar_downloader import Downloader
import pandas as pd
from bs4 import BeautifulSoup

from ..config import settings
from .sp500_collector import SP500Company

logger = logging.getLogger(__name__)


@dataclass
class EdgarFiling:
    """Data class for EDGAR filing information"""
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str
    file_path: str
    text_content: Optional[str] = None
    ma_score: Optional[float] = None  # M&A relevance score
    has_ma_content: bool = False
    processed: bool = False
    error: Optional[str] = None


class EdgarFilingCollector:
    """
    Collector for EDGAR 10-K filings from S&P 500 companies
    """

    def __init__(self):
        self.downloader = Downloader(
            company_name=settings.edgar_user_agent,
            email_address=settings.edgar_email,
            download_folder=settings.edgar_data_directory)
        self.data_directory = Path(settings.edgar_data_directory)
        self.filings_cache = Path(
            settings.cache_directory) / "edgar_filings.json"
        self.filings: List[EdgarFiling] = []

        # M&A keywords for content filtering
        self.ma_keywords = settings.ma_keywords

        # Compile regex patterns for faster matching
        self.ma_patterns = [
            re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
            for keyword in self.ma_keywords
        ]

    def download_10k_filings(
            self,
            companies: List[SP500Company],
            years: List[int],
            limit_per_company: Optional[int] = None) -> List[EdgarFiling]:
        """
        Download 10-K filings for specified companies and years
        
        Args:
            companies: List of SP500Company objects
            years: List of years to download
            limit_per_company: Optional limit on filings per company
            
        Returns:
            List of EdgarFiling objects
        """
        logger.info(
            f"Starting download of 10-K filings for {len(companies)} companies"
        )

        filings = []

        for i, company in enumerate(companies):
            if not company.cik:
                logger.warning(f"Skipping {company.symbol} - no CIK available")
                continue

            logger.info(
                f"Processing {company.symbol} ({i+1}/{len(companies)})")

            try:
                company_filings = self._download_company_filings(
                    company, years, limit_per_company)
                filings.extend(company_filings)

                # Rate limiting to be respectful to SEC servers
                time.sleep(settings.edgar_rate_limit)

            except Exception as e:
                logger.error(
                    f"Error downloading filings for {company.symbol}: {e}")
                continue

        logger.info(f"Downloaded {len(filings)} total filings")
        self.filings = filings
        return filings

    def _download_company_filings(
            self,
            company: SP500Company,
            years: List[int],
            limit: Optional[int] = None) -> List[EdgarFiling]:
        """
        Download 10-K filings for a specific company
        
        Args:
            company: SP500Company object
            years: List of years to download
            limit: Optional limit on number of filings
            
        Returns:
            List of EdgarFiling objects
        """
        filings = []

        for year in years:
            try:
                # Use sec-edgar-downloader to get filings
                # Version 5.0+ API: first parameter is form type, not keyword
                download_result = self.downloader.get(
                    "10-K",  # form type as first parameter
                    ticker_or_cik=company.cik,
                    after=f"{year}-01-01",
                    before=f"{year}-12-31",
                    limit=limit)

                # Process downloaded files
                company_folder = Path(
                    settings.edgar_data_directory
                ) / "sec-edgar-filings" / company.cik / "10-K"

                if company_folder.exists():
                    for filing_folder in company_folder.iterdir():
                        if filing_folder.is_dir():
                            filing = self._process_filing_folder(
                                company, filing_folder)
                            if filing:
                                filings.append(filing)

            except Exception as e:
                logger.warning(
                    f"Error downloading {year} filings for {company.symbol}: {e}"
                )
                continue

        return filings

    def _process_filing_folder(self, company: SP500Company,
                               filing_folder: Path) -> Optional[EdgarFiling]:
        """
        Process a downloaded filing folder
        
        Args:
            company: SP500Company object
            filing_folder: Path to the filing folder
            
        Returns:
            EdgarFiling object or None if processing failed
        """
        try:
            # Find the main filing document (usually ends with .txt)
            filing_files = list(filing_folder.glob("*.txt"))
            if not filing_files:
                logger.warning(f"No .txt files found in {filing_folder}")
                return None

            main_file = filing_files[0]  # Take the first .txt file

            # Extract metadata from folder name (accession number)
            accession_number = filing_folder.name

            # Extract filing date from the file or metadata
            filing_date = self._extract_filing_date(main_file)

            # Read the filing content
            text_content = self._extract_text_content(main_file)

            # Calculate M&A relevance score
            ma_score, has_ma_content = self._calculate_ma_score(text_content)

            filing = EdgarFiling(cik=company.cik,
                                 company_name=company.name,
                                 form_type="10-K",
                                 filing_date=filing_date,
                                 accession_number=accession_number,
                                 file_path=str(main_file),
                                 text_content=text_content,
                                 ma_score=ma_score,
                                 has_ma_content=has_ma_content,
                                 processed=True)

            return filing

        except Exception as e:
            logger.error(
                f"Error processing filing folder {filing_folder}: {e}")
            return EdgarFiling(cik=company.cik,
                               company_name=company.name,
                               form_type="10-K",
                               filing_date="",
                               accession_number=filing_folder.name,
                               file_path=str(filing_folder),
                               error=str(e),
                               processed=False)

    def _extract_filing_date(self, file_path: Path) -> str:
        """
        Extract filing date from the filing document
        
        Args:
            file_path: Path to the filing file
            
        Returns:
            Filing date as string (YYYY-MM-DD)
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5KB to find date

            # Look for FILED AS OF DATE pattern
            date_pattern = r'FILED AS OF DATE:\s*(\d{8})'
            match = re.search(date_pattern, content)

            if match:
                date_str = match.group(1)
                # Convert YYYYMMDD to YYYY-MM-DD
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # Fallback: try to extract from filename or modification date
            return datetime.fromtimestamp(
                file_path.stat().st_mtime).strftime('%Y-%m-%d')

        except Exception as e:
            logger.warning(
                f"Could not extract filing date from {file_path}: {e}")
            return ""

    def _extract_text_content(self, file_path: Path) -> str:
        """
        Extract clean text content from EDGAR filing
        
        Args:
            file_path: Path to the filing file
            
        Returns:
            Clean text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Remove EDGAR headers and formatting
            content = self._clean_edgar_content(content)

            return content

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def _clean_edgar_content(self, content: str) -> str:
        """
        Clean EDGAR filing content by removing headers, HTML tags, etc.
        
        Args:
            content: Raw filing content
            
        Returns:
            Cleaned text content
        """
        # Remove EDGAR header section
        if '<SEC-DOCUMENT>' in content:
            start_idx = content.find('<SEC-DOCUMENT>')
            content = content[start_idx:]

        # First pass: Use BeautifulSoup to extract text from HTML sections
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()
        except Exception as e:
            logger.warning(
                f"BeautifulSoup parsing failed, using fallback: {e}")
            text = content

        # Second pass: Use regex to remove any remaining HTML/XML tags
        # This catches malformed tags or tags that BeautifulSoup might miss
        text = re.sub(r'<[^>]+>', '', text)

        # Third pass: Remove HTML entities
        text = html.unescape(text)

        # Fourth pass: Remove any remaining XML/HTML-like patterns
        # Remove XBRL tags and other XML namespaced tags
        text = re.sub(r'<[^>]*?>', '', text)
        text = re.sub(r'</[^>]*?>', '', text)

        # Remove common EDGAR/XBRL artifacts
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # HTML entities
        text = re.sub(r'</?[a-zA-Z][^>]*/?>', '', text)  # Any remaining tags

        # Remove style attributes and other inline formatting
        text = re.sub(r'style="[^"]*"', '', text)
        text = re.sub(r'class="[^"]*"', '', text)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines
                  for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Final cleanup: remove any remaining angle brackets that might be artifacts
        text = re.sub(r'[<>]', '', text)

        return text

    def _calculate_ma_score(self, text: str) -> Tuple[float, bool]:
        """
        Calculate M&A relevance score for the filing text
        
        Args:
            text: Filing text content
            
        Returns:
            Tuple of (ma_score, has_ma_content)
        """
        if not text:
            return 0.0, False

        text_lower = text.lower()
        total_words = len(text.split())

        if total_words == 0:
            return 0.0, False

        # Count M&A keyword occurrences
        ma_word_count = 0
        for pattern in self.ma_patterns:
            matches = pattern.findall(text)
            ma_word_count += len(matches)

        # Calculate score as percentage of M&A words
        ma_score = (ma_word_count / total_words) * 100

        # Consider it has M&A content if score > 0.01% (1 in 10,000 words)
        has_ma_content = ma_score > 0.01

        return ma_score, has_ma_content

    def filter_ma_relevant_filings(self,
                                   min_score: float = 0.01
                                   ) -> List[EdgarFiling]:
        """
        Filter filings to only those with significant M&A content
        
        Args:
            min_score: Minimum M&A score threshold
            
        Returns:
            List of M&A relevant filings
        """
        return [
            filing for filing in self.filings
            if filing.ma_score and filing.ma_score >= min_score
        ]

    def get_filings_by_company(self, cik: str) -> List[EdgarFiling]:
        """
        Get all filings for a specific company
        
        Args:
            cik: Company CIK
            
        Returns:
            List of filings for the company
        """
        return [filing for filing in self.filings if filing.cik == cik]

    def get_filings_by_year(self, year: int) -> List[EdgarFiling]:
        """
        Get all filings for a specific year
        
        Args:
            year: Filing year
            
        Returns:
            List of filings for the year
        """
        return [
            filing for filing in self.filings
            if filing.filing_date.startswith(str(year))
        ]

    def save_cache(self):
        """Save filings to cache file"""
        try:
            cache_data = [asdict(filing) for filing in self.filings]
            with open(self.filings_cache, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(
                f"Cached {len(self.filings)} filings to {self.filings_cache}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self) -> bool:
        """
        Load filings from cache file
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if self.filings_cache.exists():
                with open(self.filings_cache, 'r') as f:
                    cached_data = json.load(f)

                self.filings = [
                    EdgarFiling(**filing) for filing in cached_data
                ]
                logger.info(f"Loaded {len(self.filings)} filings from cache")
                return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

        return False

    def test_content_cleaning(self, sample_text: str = None) -> str:
        """
        Test the content cleaning functionality with sample text
        
        Args:
            sample_text: Optional sample text to test. If None, uses a default sample.
            
        Returns:
            Cleaned text for verification
        """
        if sample_text is None:
            # Default sample with various HTML/XML tags
            sample_text = '''
            <div style="margin-top:6pt;text-align:justify;text-indent:18pt">
            <span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:120%">
            significantly, particularly in the U.S., demand for rapid COVID-19 tests increased significantly. 
            As a result, in the second half of 2021, Abbott sold approximately $181 million of inventory 
            that was previously estimated to have no net realizable value under the second quarter restructuring action.
            </span></div>
            <div style="margin-top:6pt"><table style="border-collapse:collapse;display:inline-table">
            <tr><td>Test data</td></tr></table></div>
            &nbsp;&amp;#160; HTML entities test &lt;tag&gt;
            '''

        cleaned = self._clean_edgar_content(sample_text)

        logger.info("Content cleaning test:")
        logger.info(f"Original length: {len(sample_text)}")
        logger.info(f"Cleaned length: {len(cleaned)}")
        logger.info(f"Cleaned text preview: {cleaned[:200]}...")

        return cleaned

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about collected filings
        
        Returns:
            Dictionary with summary statistics
        """
        total_filings = len(self.filings)
        processed_filings = len([f for f in self.filings if f.processed])
        ma_relevant_filings = len(
            [f for f in self.filings if f.has_ma_content])

        # Company breakdown
        company_counts = {}
        for filing in self.filings:
            company_counts[filing.company_name] = company_counts.get(
                filing.company_name, 0) + 1

        # Year breakdown
        year_counts = {}
        for filing in self.filings:
            year = filing.filing_date[:4] if filing.filing_date else "Unknown"
            year_counts[year] = year_counts.get(year, 0) + 1

        return {
            'total_filings':
            total_filings,
            'processed_filings':
            processed_filings,
            'ma_relevant_filings':
            ma_relevant_filings,
            'ma_relevance_rate':
            ma_relevant_filings / total_filings if total_filings > 0 else 0,
            'companies_count':
            len(company_counts),
            'years_covered':
            len([y for y in year_counts.keys() if y != "Unknown"]),
            'company_breakdown':
            company_counts,
            'year_breakdown':
            year_counts
        }
