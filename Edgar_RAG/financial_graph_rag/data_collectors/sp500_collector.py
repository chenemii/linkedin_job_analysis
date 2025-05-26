"""
S&P 500 Company Data Collector

Collects and maintains the current list of S&P 500 companies
along with their basic information (ticker, CIK, sector, etc.)
"""

import json
import logging
import requests
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
import time

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class SP500Company:
    """Data class for S&P 500 company information"""
    symbol: str
    name: str
    sector: str
    sub_industry: str
    headquarters: str
    date_added: str
    cik: Optional[str] = None
    founded: Optional[str] = None


class SP500Collector:
    """
    Collector for S&P 500 company data from multiple sources
    """

    def __init__(self):
        self.cache_file = Path(
            settings.cache_directory) / "sp500_companies.json"
        self.companies: List[SP500Company] = []

    def collect_from_wikipedia(self) -> List[SP500Company]:
        """
        Collect S&P 500 company list from Wikipedia
        
        Returns:
            List of SP500Company objects
        """
        logger.info("Collecting S&P 500 companies from Wikipedia")

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the main table (first table with class 'wikitable')
            table = soup.find('table', {'class': 'wikitable'})

            if not table:
                raise ValueError("Could not find S&P 500 table on Wikipedia")

            companies = []
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 7:  # Ensure we have enough columns
                    try:
                        symbol = cells[0].text.strip()
                        name = cells[1].text.strip()
                        sector = cells[2].text.strip()
                        sub_industry = cells[3].text.strip()
                        headquarters = cells[4].text.strip()
                        date_added = cells[5].text.strip()
                        founded = cells[6].text.strip() if len(
                            cells) > 6 else None

                        company = SP500Company(symbol=symbol,
                                               name=name,
                                               sector=sector,
                                               sub_industry=sub_industry,
                                               headquarters=headquarters,
                                               date_added=date_added,
                                               founded=founded)
                        companies.append(company)

                    except Exception as e:
                        logger.warning(f"Error parsing row: {e}")
                        continue

            logger.info(
                f"Collected {len(companies)} S&P 500 companies from Wikipedia")
            return companies

        except Exception as e:
            logger.error(f"Error collecting from Wikipedia: {e}")
            return []

    def enhance_with_cik_data(
            self, companies: List[SP500Company]) -> List[SP500Company]:
        """
        Enhance company data with CIK numbers from SEC
        
        Args:
            companies: List of SP500Company objects
            
        Returns:
            Enhanced list with CIK numbers
        """
        logger.info("Enhancing S&P 500 companies with CIK data")

        # First, try to fix any companies where CIK might be in the wrong field
        companies = self._fix_cik_field_issues(companies)

        # SEC company tickers endpoint
        sec_url = "https://www.sec.gov/files/company_tickers.json"

        try:
            headers = {
                'User-Agent': settings.edgar_user_agent,
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }

            response = requests.get(sec_url, headers=headers)
            response.raise_for_status()

            sec_data = response.json()

            # Create ticker to CIK mapping
            ticker_to_cik = {}
            for key, company_info in sec_data.items():
                ticker = company_info.get('ticker', '').upper()
                cik = str(company_info.get('cik_str',
                                           '')).zfill(10)  # Pad with zeros
                if ticker:
                    ticker_to_cik[ticker] = cik

            # Enhance companies with CIK data
            enhanced_companies = []
            for company in companies:
                enhanced_company = company
                cik = ticker_to_cik.get(company.symbol.upper())
                if cik:
                    enhanced_company.cik = cik

                enhanced_companies.append(enhanced_company)

            cik_found = sum(1 for c in enhanced_companies if c.cik)
            logger.info(
                f"Found CIK numbers for {cik_found}/{len(enhanced_companies)} companies"
            )

            return enhanced_companies

        except Exception as e:
            logger.error(f"Error enhancing with CIK data: {e}")
            logger.info(
                "Falling back to hardcoded CIK mappings for major companies")
            return self._add_fallback_cik_mappings(companies)

    def _fix_cik_field_issues(
            self, companies: List[SP500Company]) -> List[SP500Company]:
        """
        Fix cases where CIK numbers might be in the wrong field due to parsing issues
        """
        fixed_companies = []
        for company in companies:
            fixed_company = company
            # If CIK is null but founded field looks like a CIK (starts with zeros and is 10 digits)
            if (not company.cik and company.founded
                    and company.founded.startswith('0000')
                    and len(company.founded) == 10):
                fixed_company.cik = company.founded
                fixed_company.founded = None
                logger.debug(
                    f"Fixed CIK for {company.symbol}: moved from founded to cik field"
                )
            fixed_companies.append(fixed_company)
        return fixed_companies

    def _add_fallback_cik_mappings(
            self, companies: List[SP500Company]) -> List[SP500Company]:
        """
        Add hardcoded CIK mappings for major companies as fallback
        """
        # Hardcoded CIK mappings for major companies
        cik_mappings = {
            'AAPL': '0000320193',  # Apple
            'MSFT': '0000789019',  # Microsoft  
            'GOOGL': '0001652044',  # Alphabet Class A
            'GOOG': '0001652044',  # Alphabet Class C
            'AMZN': '0001018724',  # Amazon
            'TSLA': '0001318605',  # Tesla
            'META': '0001326801',  # Meta (Facebook)
            'NVDA': '0001045810',  # NVIDIA
            'BRK.B': '0001067983',  # Berkshire Hathaway
            'JPM': '0000019617',  # JPMorgan Chase
            'JNJ': '0000200406',  # Johnson & Johnson
            'V': '0001403161',  # Visa
            'PG': '0000080424',  # Procter & Gamble
            'HD': '0000354950',  # Home Depot
            'MA': '0001141391',  # Mastercard
            'UNH': '0000731766',  # UnitedHealth
            'DIS': '0001001039',  # Disney
            'ADBE': '0000796343',  # Adobe
            'NFLX': '0001065280',  # Netflix
            'CRM': '0001108524',  # Salesforce
            'XOM': '0000034088',  # Exxon Mobil
            'BAC': '0000070858',  # Bank of America
            'WMT': '0000104169',  # Walmart
            'KO': '0000021344',  # Coca-Cola
            'PFE': '0000078003',  # Pfizer
            'TMO': '0000097745',  # Thermo Fisher Scientific
            'CSCO': '0000858877',  # Cisco
            'ABT': '0000001800',  # Abbott Laboratories
            'CVX': '0000093410',  # Chevron
        }

        enhanced_companies = []
        cik_added = 0

        for company in companies:
            enhanced_company = company
            # If company doesn't have CIK, check our fallback mappings
            if not company.cik:
                fallback_cik = cik_mappings.get(company.symbol.upper())
                if fallback_cik:
                    enhanced_company.cik = fallback_cik
                    cik_added += 1
                    logger.debug(
                        f"Added fallback CIK for {company.symbol}: {fallback_cik}"
                    )

            enhanced_companies.append(enhanced_company)

        logger.info(f"Added fallback CIK numbers for {cik_added} companies")
        return enhanced_companies

    def collect_all(self, force_refresh: bool = False) -> List[SP500Company]:
        """
        Collect all S&P 500 company data
        
        Args:
            force_refresh: Force refresh even if cache exists
            
        Returns:
            List of SP500Company objects
        """
        # Check cache first
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)

                self.companies = [
                    SP500Company(**company) for company in cached_data
                ]
                logger.info(
                    f"Loaded {len(self.companies)} companies from cache")
                return self.companies

            except Exception as e:
                logger.warning(f"Error loading cache: {e}")

        # Collect fresh data
        logger.info("Collecting fresh S&P 500 data")

        # Get base company list from Wikipedia
        companies = self.collect_from_wikipedia()

        if not companies:
            logger.error("Failed to collect company data")
            return []

        # Enhance with CIK data
        companies = self.enhance_with_cik_data(companies)

        # Cache the results
        self._cache_companies(companies)

        self.companies = companies
        return companies

    def _cache_companies(self, companies: List[SP500Company]):
        """Cache company data to JSON file"""
        try:
            cache_data = [asdict(company) for company in companies]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(
                f"Cached {len(companies)} companies to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error caching companies: {e}")

    def get_companies_with_cik(self) -> List[SP500Company]:
        """
        Get only companies that have CIK numbers (needed for EDGAR filings)
        
        Returns:
            List of companies with CIK numbers
        """
        return [company for company in self.companies if company.cik]

    def get_company_by_symbol(self, symbol: str) -> Optional[SP500Company]:
        """
        Get company by ticker symbol
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            SP500Company object or None if not found
        """
        # Auto-load companies if not already loaded
        if not self.companies:
            self.collect_all()

        for company in self.companies:
            if company.symbol.upper() == symbol.upper():
                return company
        return None

    def get_companies_by_sector(self, sector: str) -> List[SP500Company]:
        """
        Get companies by sector
        
        Args:
            sector: Sector name
            
        Returns:
            List of companies in the sector
        """
        return [
            company for company in self.companies if company.sector == sector
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert companies to pandas DataFrame
        
        Returns:
            DataFrame with company data
        """
        return pd.DataFrame([asdict(company) for company in self.companies])

    def get_sectors(self) -> List[str]:
        """Get unique sectors"""
        return list(set(company.sector for company in self.companies))

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the S&P 500 companies
        
        Returns:
            Dictionary with summary statistics
        """
        # Auto-load companies if not already loaded
        if not self.companies:
            self.collect_all()

        stats = {
            'total_companies': len(self.companies),
            'companies_with_cik': len(self.get_companies_with_cik()),
            'sectors': len(self.get_sectors()),
            'sector_breakdown': {}
        }

        # Sector breakdown
        for sector in self.get_sectors():
            companies_in_sector = self.get_companies_by_sector(sector)
            stats['sector_breakdown'][sector] = len(companies_in_sector)

        return stats
