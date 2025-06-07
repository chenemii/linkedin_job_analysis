"""
Skill Shortage Analyzer

Analyzes 10-K filings for skill shortage and talent gap mentions,
integrating with the existing Financial Graph RAG system.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import re

import pandas as pd
import requests

from ..config import settings
from .sp500_collector import SP500Company
from .edgar_collector import EdgarFiling

logger = logging.getLogger(__name__)


@dataclass
class SkillShortageAnalysis:
    """Data class for skill shortage analysis results"""
    cik: str
    company_name: str
    ticker: str
    year: int
    filing_url: str
    skill_shortage_mentions: int
    skill_shortage_score: float
    analyzed_date: str
    gvkey: Optional[str] = None
    error: Optional[str] = None


class SkillShortageAnalyzer:
    """
    Analyzer for skill shortage mentions in 10-K filings
    """

    def __init__(self):
        self.skill_shortage_terms = [
            "skills gap",
            "shortage of skilled labor",
            "lack of qualified personnel",
            "difficulty attracting skilled employees",
            "competition for skilled talent",
            "challenges in recruiting skilled professionals",
            "difficulty in hiring qualified employees",
            "inability to find qualified candidates",
            "demand for skilled labor exceeds supply",
            "inadequate availability of skilled workers",
            "difficulty in filling critical roles",
            "shortage of qualified workers",
            "difficulty hiring experienced personnel",
            "high demand for specialized talent",
            "limited pool of qualified candidates",
            "shortage of experienced professionals",
            "lack of specialized skills",
            "talent shortage",
            "need for skilled labor",
            "shortage of talent",
            "recruitment challenges",
            "inability to retain skilled talent",
            "difficulty maintaining workforce with required skills",
            "difficulty retaining top talent",
            "skills shortage",
            "lack of necessary expertise",
            "scarcity of talent",
            "challenges in attracting experienced professionals",
            "lack of workforce with technical skills",
            "insufficient supply of skilled workers",
            "demand for talent in specialized areas",
            "difficulty finding qualified employees",
            "competition for qualified professionals",
            "insufficient skilled workforce",
            "workforce skills gap",
            "difficulty in filling technical roles",
            "inability to hire employees with the right skills",
            "difficulty in hiring specialized personnel",
            "workforce challenges",
            "gaps in technical expertise",
            "lack of skilled professionals",
            "limited talent availability",
            "insufficient labor force with required skills",
            "lack of highly qualified candidates",
            "workforce limitations in critical areas",
            "difficulty in hiring and retaining skilled labor",
            "insufficient number of qualified workers"
        ]
        
        # Compile regex patterns for faster matching
        self.skill_shortage_patterns = [
            re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            for term in self.skill_shortage_terms
        ]
        
        self.cache_file = Path(settings.cache_directory) / "skill_shortage_analysis.json"
        self.results: List[SkillShortageAnalysis] = []

    def count_skill_shortage_keywords(self, text: str) -> int:
        """
        Count skill shortage keyword occurrences in text
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of skill shortage keyword mentions
        """
        if not text:
            return 0
            
        text_lower = text.lower()
        total_count = 0
        
        for pattern in self.skill_shortage_patterns:
            matches = pattern.findall(text)
            total_count += len(matches)
            
        return total_count

    def calculate_skill_shortage_score(self, text: str) -> Tuple[int, float]:
        """
        Calculate skill shortage score for the text
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (mention_count, score_percentage)
        """
        if not text:
            return 0, 0.0
            
        mention_count = self.count_skill_shortage_keywords(text)
        total_words = len(text.split())
        
        if total_words == 0:
            return mention_count, 0.0
            
        # Calculate score as percentage of total words
        score = (mention_count / total_words) * 100
        
        return mention_count, score

    def analyze_edgar_filings(self, 
                            filings: List[EdgarFiling],
                            companies: List[SP500Company] = None) -> List[SkillShortageAnalysis]:
        """
        Analyze EDGAR filings for skill shortage mentions
        
        Args:
            filings: List of EDGAR filings to analyze
            companies: Optional list of companies for additional metadata
            
        Returns:
            List of skill shortage analysis results
        """
        logger.info(f"Analyzing {len(filings)} EDGAR filings for skill shortage mentions")
        
        results = []
        company_lookup = {}
        
        # Create company lookup if provided
        if companies:
            company_lookup = {company.cik: company for company in companies}
        
        for i, filing in enumerate(filings):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(filings)} filings")
                
            try:
                if not filing.text_content:
                    logger.warning(f"No text content for filing {filing.accession_number}")
                    continue
                
                # Analyze skill shortage content
                mention_count, score = self.calculate_skill_shortage_score(filing.text_content)
                
                # Get company info
                company = company_lookup.get(filing.cik)
                ticker = company.symbol if company else filing.cik
                
                # Extract year from filing date
                year = int(filing.filing_date[:4]) if filing.filing_date else None
                
                analysis = SkillShortageAnalysis(
                    cik=filing.cik,
                    company_name=filing.company_name,
                    ticker=ticker,
                    year=year,
                    filing_url=f"https://www.sec.gov/Archives/{filing.file_path}" if hasattr(filing, 'file_path') else "",
                    skill_shortage_mentions=mention_count,
                    skill_shortage_score=score,
                    analyzed_date=pd.Timestamp.now().isoformat(),
                    gvkey=getattr(company, 'gvkey', None) if company else None
                )
                
                results.append(analysis)
                
                # Log significant findings
                if mention_count > 0:
                    logger.info(f"Found {mention_count} skill shortage mentions in {ticker} ({year})")
                    
            except Exception as e:
                logger.error(f"Error analyzing filing {filing.accession_number}: {e}")
                continue
        
        self.results = results
        logger.info(f"Completed analysis. Found skill shortage mentions in {len([r for r in results if r.skill_shortage_mentions > 0])} filings")
        
        return results

    def analyze_from_urls(self, 
                         df: pd.DataFrame,
                         limit: Optional[int] = None) -> List[SkillShortageAnalysis]:
        """
        Analyze 10-K filings from URLs (similar to the original provided code)
        
        Args:
            df: DataFrame with columns: cik, Year, FName, gvkey
            limit: Optional limit on number of filings to process
            
        Returns:
            List of skill shortage analysis results
        """
        logger.info(f"Analyzing filings from {len(df)} URLs")
        
        results = []
        base_url = "https://www.sec.gov/Archives/"
        headers = {"User-Agent": f"{settings.edgar_user_agent} ({settings.edgar_email})"}
        
        sample_df = df.copy()
        if limit:
            sample_df = sample_df.head(limit)
        
        for index, row in sample_df.iterrows():
            try:
                url = base_url + row['FName']
                
                # Rate limiting
                time.sleep(settings.edgar_rate_limit)
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    text = response.text
                    mention_count, score = self.calculate_skill_shortage_score(text)
                    
                    if mention_count > 0:
                        logger.info(f"Found {mention_count} mentions for CIK {row['cik']} ({row['Year']})")
                    
                    analysis = SkillShortageAnalysis(
                        cik=str(row['cik']),
                        company_name=f"Company_{row['cik']}",  # Would need company name lookup
                        ticker=str(row['cik']),  # Would need ticker lookup
                        year=int(row['Year']),
                        filing_url=url,
                        skill_shortage_mentions=mention_count,
                        skill_shortage_score=score,
                        analyzed_date=pd.Timestamp.now().isoformat(),
                        gvkey=str(row.get('gvkey', '')) if 'gvkey' in row else None
                    )
                    
                    results.append(analysis)
                    
                else:
                    logger.warning(f"Failed to fetch {url}: Status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                continue
        
        self.results = results
        return results

    def filter_significant_findings(self, 
                                  min_mentions: int = 1,
                                  min_score: float = 0.001) -> List[SkillShortageAnalysis]:
        """
        Filter results to only significant skill shortage findings
        
        Args:
            min_mentions: Minimum number of mentions
            min_score: Minimum skill shortage score
            
        Returns:
            Filtered list of significant findings
        """
        return [
            result for result in self.results
            if result.skill_shortage_mentions >= min_mentions and result.skill_shortage_score >= min_score
        ]

    def get_company_skill_shortage_summary(self, cik: str) -> Dict:
        """
        Get skill shortage summary for a specific company
        
        Args:
            cik: Company CIK
            
        Returns:
            Summary dictionary
        """
        company_results = [r for r in self.results if r.cik == cik]
        
        if not company_results:
            return {"error": f"No results found for CIK {cik}"}
        
        total_mentions = sum(r.skill_shortage_mentions for r in company_results)
        avg_score = sum(r.skill_shortage_score for r in company_results) / len(company_results)
        years_analyzed = sorted(list(set(r.year for r in company_results if r.year)))
        
        return {
            "cik": cik,
            "company_name": company_results[0].company_name,
            "ticker": company_results[0].ticker,
            "total_mentions": total_mentions,
            "average_score": avg_score,
            "years_analyzed": years_analyzed,
            "filings_count": len(company_results),
            "years_with_mentions": len([r for r in company_results if r.skill_shortage_mentions > 0])
        }

    def get_industry_trends(self, companies: List[SP500Company] = None) -> Dict:
        """
        Analyze skill shortage trends by industry/sector
        
        Args:
            companies: List of companies for sector information
            
        Returns:
            Industry trends analysis
        """
        if not companies:
            return {"error": "Company information required for industry analysis"}
        
        # Create company lookup
        company_lookup = {company.cik: company for company in companies}
        
        # Group results by sector
        sector_data = {}
        
        for result in self.results:
            company = company_lookup.get(result.cik)
            if not company:
                continue
                
            sector = company.sector
            if sector not in sector_data:
                sector_data[sector] = {
                    "companies": set(),
                    "total_mentions": 0,
                    "total_filings": 0,
                    "companies_with_mentions": set()
                }
            
            sector_data[sector]["companies"].add(result.cik)
            sector_data[sector]["total_mentions"] += result.skill_shortage_mentions
            sector_data[sector]["total_filings"] += 1
            
            if result.skill_shortage_mentions > 0:
                sector_data[sector]["companies_with_mentions"].add(result.cik)
        
        # Calculate sector statistics
        sector_stats = {}
        for sector, data in sector_data.items():
            sector_stats[sector] = {
                "companies_count": len(data["companies"]),
                "total_mentions": data["total_mentions"],
                "total_filings": data["total_filings"],
                "companies_with_mentions": len(data["companies_with_mentions"]),
                "mention_rate": len(data["companies_with_mentions"]) / len(data["companies"]) if data["companies"] else 0,
                "avg_mentions_per_filing": data["total_mentions"] / data["total_filings"] if data["total_filings"] else 0
            }
        
        return sector_stats

    def save_results(self, output_path: str = None) -> str:
        """
        Save analysis results to CSV file
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if not output_path:
            output_path = Path(settings.data_directory) / "skill_shortage_analysis.csv"
        
        # Convert results to DataFrame
        results_data = [asdict(result) for result in self.results]
        df = pd.DataFrame(results_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(self.results)} results to {output_path}")
        
        return str(output_path)

    def save_cache(self):
        """Save results to cache file"""
        try:
            cache_data = [asdict(result) for result in self.results]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached {len(self.results)} results to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self) -> bool:
        """
        Load results from cache file
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                self.results = [
                    SkillShortageAnalysis(**result) for result in cached_data
                ]
                logger.info(f"Loaded {len(self.results)} results from cache")
                return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
        
        return False

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the analysis
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No analysis results available"}
        
        total_filings = len(self.results)
        filings_with_mentions = len([r for r in self.results if r.skill_shortage_mentions > 0])
        total_mentions = sum(r.skill_shortage_mentions for r in self.results)
        
        # Company and year breakdowns
        companies = set(r.cik for r in self.results)
        years = set(r.year for r in self.results if r.year)
        
        # Top companies by mentions
        company_mentions = {}
        for result in self.results:
            key = f"{result.ticker} ({result.cik})"
            company_mentions[key] = company_mentions.get(key, 0) + result.skill_shortage_mentions
        
        top_companies = sorted(company_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_filings_analyzed": total_filings,
            "filings_with_skill_shortage_mentions": filings_with_mentions,
            "skill_shortage_mention_rate": filings_with_mentions / total_filings if total_filings > 0 else 0,
            "total_skill_shortage_mentions": total_mentions,
            "average_mentions_per_filing": total_mentions / total_filings if total_filings > 0 else 0,
            "unique_companies": len(companies),
            "years_covered": sorted(list(years)),
            "top_companies_by_mentions": top_companies
        } 