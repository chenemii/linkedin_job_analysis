"""
Skill Shortage Analyzer

Analyzes 10-K filings for skill shortage and talent gap mentions,
integrating with the existing Financial Graph RAG system.
"""

import logging
import time
import os
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import re

import pandas as pd
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..config import settings
from .sp500_collector import SP500Company
from .edgar_collector import EdgarFiling

logger = logging.getLogger(__name__)

# Fix tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log_llm_request(llm, messages, context=""):
    """
    Wrapper function to log LLM requests with full URL for debugging
    """
    # Try multiple ways to get the base URL
    base_url = None
    model_name = getattr(llm, 'model_name', 'unknown')
    
    # Method 1: Check if it's a ChatOpenAI instance with client
    if hasattr(llm, 'client') and hasattr(llm.client, 'base_url'):
        base_url = str(llm.client.base_url)
    
    # Method 2: Check for openai_api_base attribute
    elif hasattr(llm, 'openai_api_base'):
        base_url = str(llm.openai_api_base)
    
    # Method 3: Check for base_url attribute directly
    elif hasattr(llm, 'base_url'):
        base_url = str(llm.base_url)
    
    # Method 4: Check client._base_url (newer OpenAI client versions)
    elif hasattr(llm, 'client') and hasattr(llm.client, '_base_url'):
        base_url = str(llm.client._base_url)
    
    # Method 5: Check for default OpenAI endpoint
    elif 'openai' in str(type(llm)).lower():
        # Default OpenAI endpoint
        base_url = "https://api.openai.com/v1"
    
    if base_url:
        # Ensure the URL ends with the chat completions endpoint
        if not base_url.endswith('/chat/completions'):
            if base_url.endswith('/v1'):
                full_url = f"{base_url}/chat/completions"
            else:
                full_url = f"{base_url}/v1/chat/completions"
        else:
            full_url = base_url
            
        logger.info(f"Making LLM request {context} to: {full_url}")
        logger.info(f"Model: {model_name}")
    else:
        logger.info(f"Making LLM request {context} (URL detection failed)")
        logger.info(f"Model: {model_name}")
        logger.info(f"LLM type: {type(llm)}")
    
    return llm.invoke(messages)

def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate delay with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    
    raise Exception(f"Failed after {max_retries} attempts")

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
    skill_shortage_likelihood: float  # New AI-based likelihood score (0-10)
    skill_shortage_summary: str  # New AI-generated summary
    analyzed_date: str
    gvkey: Optional[str] = None
    error: Optional[str] = None


class SkillShortageAnalyzer:
    """
    Analyzer for skill shortage mentions in 10-K filings with AI-based likelihood scoring
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
        
        # Delay LLM initialization to avoid fork issues
        self._llm = None
        self._likelihood_prompt = None
        
        self.cache_file = Path(settings.cache_directory) / "skill_shortage_analysis.json"
        self.results: List[SkillShortageAnalysis] = []

    @property
    def llm(self):
        """Lazy load LLM to avoid fork issues"""
        if self._llm is None:
            logger.info("Initializing LLM for skill shortage analysis...")
            if settings.default_llm_provider == "openai":
                # Log the full URL for debugging
                logger.info(f"Initializing OpenAI client with base_url: {settings.openai_base_url}")
                logger.info(f"Full chat completions URL: {settings.openai_base_url}/chat/completions")
                
                self._llm = ChatOpenAI(
                    model=settings.default_llm_model,
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    timeout=30,
                    max_retries=3
                )
            else:
                # Fallback to default OpenAI for now
                logger.info("Using fallback OpenAI client with default base_url")
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=30,
                    max_retries=3
                )
            logger.info("LLM initialized successfully")
        return self._llm

    @property
    def likelihood_prompt(self):
        """Lazy load likelihood prompt to avoid fork issues"""
        if self._likelihood_prompt is None:
            self._likelihood_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert analyst specializing in workforce challenges and skill shortages in corporate environments. 

Your task is to analyze SEC filing text and determine the likelihood that a company is facing skill shortages based on the language, context, and patterns in the text - NOT just keyword matching.

Consider these factors:
1. Language patterns indicating workforce challenges
2. Context around hiring, recruitment, and talent acquisition
3. Mentions of competitive pressures for talent
4. Training and development initiatives as responses to gaps
5. Business impact statements related to human capital
6. Strategic concerns about workforce capabilities
7. Industry-specific talent challenges
8. Geographic or market-specific hiring difficulties

Provide a numerical likelihood score from 0-10 where:
- 0-2: Very low likelihood of skill shortages
- 3-4: Low likelihood 
- 5-6: Moderate likelihood
- 7-8: High likelihood
- 9-10: Very high likelihood of significant skill shortages

Focus on the overall narrative and business context, not just explicit mentions of skill shortage terms."""),
                ("human", """Analyze the following text from a 10-K filing and determine the likelihood (0-10) that this company is facing skill shortages.

Text to analyze:
{text}

Provide your response in the following JSON format:
{{
    "likelihood_score": <score_0_to_10>,
    "summary": "<brief_summary_of_key_findings>",
    "key_indicators": ["<indicator1>", "<indicator2>", "<indicator3>"],
    "confidence": "<high/medium/low>"
}}""")
            ])
        return self._likelihood_prompt

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
        Calculate skill shortage score for the text (legacy keyword-based method)
        
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

    def calculate_skill_shortage_likelihood(self, text: str) -> Tuple[float, str]:
        """
        Calculate AI-based skill shortage likelihood score
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (likelihood_score_0_to_10, summary_text)
        """
        if not text or len(text.strip()) < 100:
            return 0.0, "Insufficient text for analysis"
        
        def ai_analysis():
            # Truncate text if too long to avoid token limits
            max_chars = 8000  # Roughly 2000 tokens
            if len(text) > max_chars:
                # Take first part and last part to capture both context
                text_truncated = text[:max_chars//2] + "\n...[content truncated]...\n" + text[-max_chars//2:]
            else:
                text_truncated = text
            
            # Get AI analysis
            messages = self.likelihood_prompt.format_messages(text=text_truncated)
            response = log_llm_request(self.llm, messages, "for skill shortage likelihood analysis")
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                likelihood_score = float(result.get('likelihood_score', 0))
                summary = result.get('summary', 'No summary available')
                
                # Ensure score is within bounds
                likelihood_score = max(0.0, min(10.0, likelihood_score))
                
                return likelihood_score, summary
                
            except json.JSONDecodeError:
                # Fallback: try to extract score from text response
                import re
                score_match = re.search(r'"likelihood_score":\s*(\d+(?:\.\d+)?)', response.content)
                if score_match:
                    likelihood_score = float(score_match.group(1))
                    likelihood_score = max(0.0, min(10.0, likelihood_score))
                    return likelihood_score, "AI analysis completed (partial parsing)"
                else:
                    logger.warning("Could not parse AI response, using fallback")
                    return 0.0, "AI analysis failed - parsing error"
        
        try:
            # Use retry logic for AI analysis
            return retry_with_backoff(ai_analysis, max_retries=3, base_delay=2.0)
                    
        except Exception as e:
            logger.error(f"Error in AI likelihood calculation: {e}")
            return 0.0, f"AI analysis failed: {str(e)}"

    def analyze_edgar_filings(self, 
                            filings: List[EdgarFiling],
                            companies: List[SP500Company] = None) -> List[SkillShortageAnalysis]:
        """
        Analyze EDGAR filings for skill shortage mentions with AI-based likelihood scoring
        
        Args:
            filings: List of EDGAR filings to analyze
            companies: Optional list of companies for additional metadata
            
        Returns:
            List of skill shortage analysis results
        """
        logger.info(f"Analyzing {len(filings)} EDGAR filings for skill shortage likelihood")
        
        results = []
        company_lookup = {}
        
        # Create company lookup if provided
        if companies:
            company_lookup = {company.cik: company for company in companies}
        
        for i, filing in enumerate(filings):
            if i % 50 == 0:  # Reduced frequency due to AI processing time
                logger.info(f"Processed {i}/{len(filings)} filings")
                
            try:
                if not filing.text_content:
                    logger.warning(f"No text content for filing {filing.accession_number}")
                    continue
                
                # Calculate both legacy keyword score and new AI likelihood
                mention_count, keyword_score = self.calculate_skill_shortage_score(filing.text_content)
                likelihood_score, summary = self.calculate_skill_shortage_likelihood(filing.text_content)
                
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
                    skill_shortage_score=keyword_score,
                    skill_shortage_likelihood=likelihood_score,
                    skill_shortage_summary=summary,
                    analyzed_date=pd.Timestamp.now().isoformat(),
                    gvkey=getattr(company, 'gvkey', None) if company else None
                )
                
                results.append(analysis)
                
                # Log significant findings
                if likelihood_score > 5.0 or mention_count > 0:
                    logger.info(f"Found skill shortage likelihood {likelihood_score:.1f}/10 for {ticker} ({year}) - {mention_count} keyword mentions")
                    
            except Exception as e:
                logger.error(f"Error analyzing filing {filing.accession_number}: {e}")
                continue
        
        self.results = results
        logger.info(f"Completed analysis. Found {len([r for r in results if r.skill_shortage_likelihood > 5.0])} filings with high skill shortage likelihood (>5.0)")
        
        return results

    def analyze_from_urls(self, 
                         df: pd.DataFrame,
                         limit: Optional[int] = None) -> List[SkillShortageAnalysis]:
        """
        Analyze 10-K filings from URLs with AI-based likelihood scoring
        
        Args:
            df: DataFrame with columns: cik, Year, FName, gvkey
            limit: Optional limit on number of filings to process
            
        Returns:
            List of skill shortage analysis results
        """
        logger.info(f"Analyzing filings from {len(df)} URLs with AI likelihood scoring")
        
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
                    
                    # Calculate both legacy keyword score and new AI likelihood
                    mention_count, keyword_score = self.calculate_skill_shortage_score(text)
                    likelihood_score, summary = self.calculate_skill_shortage_likelihood(text)
                    
                    if likelihood_score > 3.0 or mention_count > 0:
                        logger.info(f"Found skill shortage likelihood {likelihood_score:.1f}/10 for CIK {row['cik']} ({row['Year']}) - {mention_count} keyword mentions")
                    
                    analysis = SkillShortageAnalysis(
                        cik=str(row['cik']),
                        company_name=f"Company_{row['cik']}",  # Would need company name lookup
                        ticker=str(row['cik']),  # Would need ticker lookup
                        year=int(row['Year']),
                        filing_url=url,
                        skill_shortage_mentions=mention_count,
                        skill_shortage_score=keyword_score,
                        skill_shortage_likelihood=likelihood_score,
                        skill_shortage_summary=summary,
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
                                  min_score: float = 0.001,
                                  min_likelihood: float = 5.0) -> List[SkillShortageAnalysis]:
        """
        Filter results to only significant skill shortage findings
        
        Args:
            min_mentions: Minimum number of keyword mentions
            min_score: Minimum keyword-based score
            min_likelihood: Minimum AI likelihood score (0-10)
            
        Returns:
            Filtered list of significant findings
        """
        return [
            result for result in self.results
            if (result.skill_shortage_mentions >= min_mentions and result.skill_shortage_score >= min_score) 
            or result.skill_shortage_likelihood >= min_likelihood
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
        avg_keyword_score = sum(r.skill_shortage_score for r in company_results) / len(company_results)
        avg_likelihood_score = sum(r.skill_shortage_likelihood for r in company_results) / len(company_results)
        years_analyzed = sorted(list(set(r.year for r in company_results if r.year)))
        
        # Get the most recent summary
        recent_summary = ""
        if company_results:
            recent_result = max(company_results, key=lambda x: x.year if x.year else 0)
            recent_summary = recent_result.skill_shortage_summary
        
        return {
            "cik": cik,
            "company_name": company_results[0].company_name,
            "ticker": company_results[0].ticker,
            "total_mentions": total_mentions,
            "average_keyword_score": avg_keyword_score,
            "average_likelihood_score": avg_likelihood_score,
            "recent_summary": recent_summary,
            "years_analyzed": years_analyzed,
            "filings_count": len(company_results),
            "years_with_mentions": len([r for r in company_results if r.skill_shortage_mentions > 0]),
            "years_with_high_likelihood": len([r for r in company_results if r.skill_shortage_likelihood > 6.0])
        }

    def get_industry_trends(self, companies: List[SP500Company] = None) -> Dict:
        """
        Analyze skill shortage trends by industry/sector with AI likelihood scores
        
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
                    "total_likelihood_score": 0,
                    "total_filings": 0,
                    "companies_with_mentions": set(),
                    "companies_with_high_likelihood": set()
                }
            
            sector_data[sector]["companies"].add(result.cik)
            sector_data[sector]["total_mentions"] += result.skill_shortage_mentions
            sector_data[sector]["total_likelihood_score"] += result.skill_shortage_likelihood
            sector_data[sector]["total_filings"] += 1
            
            if result.skill_shortage_mentions > 0:
                sector_data[sector]["companies_with_mentions"].add(result.cik)
            
            if result.skill_shortage_likelihood > 6.0:
                sector_data[sector]["companies_with_high_likelihood"].add(result.cik)
        
        # Calculate sector statistics
        sector_stats = {}
        for sector, data in sector_data.items():
            sector_stats[sector] = {
                "companies_count": len(data["companies"]),
                "total_mentions": data["total_mentions"],
                "avg_likelihood_score": data["total_likelihood_score"] / data["total_filings"] if data["total_filings"] else 0,
                "total_filings": data["total_filings"],
                "companies_with_mentions": len(data["companies_with_mentions"]),
                "companies_with_high_likelihood": len(data["companies_with_high_likelihood"]),
                "mention_rate": len(data["companies_with_mentions"]) / len(data["companies"]) if data["companies"] else 0,
                "high_likelihood_rate": len(data["companies_with_high_likelihood"]) / len(data["companies"]) if data["companies"] else 0,
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
        Get summary statistics about the analysis including AI likelihood scores
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No analysis results available"}
        
        total_filings = len(self.results)
        filings_with_mentions = len([r for r in self.results if r.skill_shortage_mentions > 0])
        filings_with_high_likelihood = len([r for r in self.results if r.skill_shortage_likelihood > 6.0])
        total_mentions = sum(r.skill_shortage_mentions for r in self.results)
        
        # Calculate average likelihood score
        avg_likelihood = sum(r.skill_shortage_likelihood for r in self.results) / total_filings if total_filings > 0 else 0
        
        # Company and year breakdowns
        companies = set(r.cik for r in self.results)
        years = set(r.year for r in self.results if r.year)
        
        # Top companies by likelihood score
        company_likelihood = {}
        for result in self.results:
            key = f"{result.ticker} ({result.cik})"
            if key not in company_likelihood:
                company_likelihood[key] = []
            company_likelihood[key].append(result.skill_shortage_likelihood)
        
        # Calculate average likelihood per company
        company_avg_likelihood = {
            company: sum(scores) / len(scores) 
            for company, scores in company_likelihood.items()
        }
        
        top_companies_by_likelihood = sorted(
            company_avg_likelihood.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Top companies by keyword mentions (legacy)
        company_mentions = {}
        for result in self.results:
            key = f"{result.ticker} ({result.cik})"
            company_mentions[key] = company_mentions.get(key, 0) + result.skill_shortage_mentions
        
        top_companies_by_mentions = sorted(company_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_filings_analyzed": total_filings,
            "filings_with_skill_shortage_mentions": filings_with_mentions,
            "filings_with_high_likelihood": filings_with_high_likelihood,
            "skill_shortage_mention_rate": filings_with_mentions / total_filings if total_filings > 0 else 0,
            "high_likelihood_rate": filings_with_high_likelihood / total_filings if total_filings > 0 else 0,
            "total_skill_shortage_mentions": total_mentions,
            "average_mentions_per_filing": total_mentions / total_filings if total_filings > 0 else 0,
            "average_likelihood_score": avg_likelihood,
            "unique_companies": len(companies),
            "years_covered": sorted(list(years)),
            "top_companies_by_likelihood": top_companies_by_likelihood,
            "top_companies_by_mentions": top_companies_by_mentions
        }

    def rank_companies_by_skill_shortage(self, 
                                       min_filings: int = 1,
                                       recent_years_only: bool = False,
                                       years_lookback: int = 3) -> List[Dict]:
        """
        Rank all companies by skill shortage levels using similarity-based scoring
        
        Args:
            min_filings: Minimum number of filings required for a company to be included
            recent_years_only: If True, only consider filings from recent years
            years_lookback: Number of recent years to consider if recent_years_only is True
            
        Returns:
            List of dictionaries with company rankings, sorted from highest to lowest skill shortage
        """
        if not self.results:
            logger.warning("No analysis results available for ranking")
            return []
        
        # Filter results by recent years if requested
        filtered_results = self.results
        if recent_years_only:
            current_year = pd.Timestamp.now().year
            cutoff_year = current_year - years_lookback
            filtered_results = [r for r in self.results if r.year and r.year >= cutoff_year]
            logger.info(f"Filtering to filings from {cutoff_year} onwards: {len(filtered_results)} filings")
        
        # Group results by company
        company_data = {}
        for result in filtered_results:
            company_key = result.cik
            
            if company_key not in company_data:
                company_data[company_key] = {
                    "cik": result.cik,
                    "company_name": result.company_name,
                    "ticker": result.ticker,
                    "gvkey": result.gvkey,
                    "filings": [],
                    "total_mentions": 0,
                    "total_likelihood_score": 0,
                    "max_likelihood_score": 0,
                    "years_analyzed": set(),
                    "filings_with_mentions": 0,
                    "filings_with_high_likelihood": 0,
                    "recent_summaries": []
                }
            
            company_data[company_key]["filings"].append(result)
            company_data[company_key]["total_mentions"] += result.skill_shortage_mentions
            company_data[company_key]["total_likelihood_score"] += result.skill_shortage_likelihood
            company_data[company_key]["max_likelihood_score"] = max(
                company_data[company_key]["max_likelihood_score"], 
                result.skill_shortage_likelihood
            )
            
            if result.year:
                company_data[company_key]["years_analyzed"].add(result.year)
            
            if result.skill_shortage_mentions > 0:
                company_data[company_key]["filings_with_mentions"] += 1
            
            if result.skill_shortage_likelihood > 6.0:
                company_data[company_key]["filings_with_high_likelihood"] += 1
            
            # Collect recent summaries (non-empty ones)
            if result.skill_shortage_summary and result.skill_shortage_summary.strip() and len(result.skill_shortage_summary) > 20:
                company_data[company_key]["recent_summaries"].append({
                    "year": result.year,
                    "summary": result.skill_shortage_summary,
                    "likelihood": result.skill_shortage_likelihood
                })
        
        # Filter companies by minimum filings requirement
        company_data = {k: v for k, v in company_data.items() if len(v["filings"]) >= min_filings}
        
        if not company_data:
            logger.warning("No companies meet the minimum filings requirement")
            return []
        
        # Calculate similarity-based ranking score for each company
        company_rankings = []
        for company_key, data in company_data.items():
            filings_count = len(data["filings"])
            avg_likelihood_score = data["total_likelihood_score"] / filings_count
            avg_mentions_per_filing = data["total_mentions"] / filings_count
            mention_rate = data["filings_with_mentions"] / filings_count
            high_likelihood_rate = data["filings_with_high_likelihood"] / filings_count
            
            # Get most recent and highest likelihood summary
            recent_summary = ""
            if data["recent_summaries"]:
                # Sort by likelihood score first, then by year
                sorted_summaries = sorted(data["recent_summaries"], 
                                        key=lambda x: (x["likelihood"], x["year"] or 0), 
                                        reverse=True)
                recent_summary = sorted_summaries[0]["summary"]
            
            # Similarity-based scoring: combines AI likelihood with mention patterns
            # This creates a score that reflects how similar the company's filings are to skill shortage patterns
            similarity_score = (
                avg_likelihood_score * 0.6 +  # Primary weight on AI likelihood (semantic similarity)
                data["max_likelihood_score"] * 0.2 +  # Peak similarity score
                mention_rate * 3.0 +  # Consistency of mentions across filings
                high_likelihood_rate * 2.0  # Proportion of high-confidence matches
            )
            
            company_ranking = {
                "rank": 0,  # Will be set after sorting
                "cik": data["cik"],
                "company_name": data["company_name"],
                "ticker": data["ticker"],
                "gvkey": data["gvkey"],
                "filings_analyzed": filings_count,
                "years_analyzed": sorted(list(data["years_analyzed"])),
                "total_mentions": data["total_mentions"],
                "avg_mentions_per_filing": round(avg_mentions_per_filing, 2),
                "avg_likelihood_score": round(avg_likelihood_score, 2),
                "max_likelihood_score": round(data["max_likelihood_score"], 2),
                "mention_rate": round(mention_rate, 3),
                "high_likelihood_rate": round(high_likelihood_rate, 3),
                "filings_with_mentions": data["filings_with_mentions"],
                "filings_with_high_likelihood": data["filings_with_high_likelihood"],
                "similarity_score": round(similarity_score, 2),
                "recent_summary": recent_summary
            }
            
            company_rankings.append(company_ranking)
        
        # Sort by similarity score (highest first)
        company_rankings.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Assign ranks
        for i, company in enumerate(company_rankings):
            company["rank"] = i + 1
        
        logger.info(f"Ranked {len(company_rankings)} companies by skill shortage similarity")
        
        # Log top 10 companies
        top_10 = company_rankings[:10]
        logger.info("Top 10 companies with highest skill shortage similarity:")
        for company in top_10:
            logger.info(f"  {company['rank']}. {company['ticker']} - {company['company_name']} (Similarity Score: {company['similarity_score']})")
        
        return company_rankings

    def export_company_rankings(self, 
                              rankings: List[Dict] = None,
                              output_path: str = None,
                              include_summaries: bool = True) -> str:
        """
        Export company rankings to CSV file
        
        Args:
            rankings: List of company rankings (if None, will generate using default method)
            output_path: Optional custom output path
            include_summaries: Whether to include AI-generated summaries in export
            
        Returns:
            Path to exported file
        """
        if rankings is None:
            rankings = self.rank_companies_by_skill_shortage()
        
        if not rankings:
            raise ValueError("No rankings available to export")
        
        if not output_path:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(settings.data_directory) / f"skill_shortage_company_rankings_{timestamp}.csv"
        
        # Prepare data for export
        export_data = []
        for company in rankings:
            row = {
                "rank": company["rank"],
                "ticker": company["ticker"],
                "company_name": company["company_name"],
                "cik": company["cik"],
                "gvkey": company["gvkey"],
                "filings_analyzed": company["filings_analyzed"],
                "years_analyzed": ", ".join(map(str, company["years_analyzed"])),
                "avg_likelihood_score": company["avg_likelihood_score"],
                "max_likelihood_score": company["max_likelihood_score"],
                "total_mentions": company["total_mentions"],
                "avg_mentions_per_filing": company["avg_mentions_per_filing"],
                "mention_rate": company["mention_rate"],
                "high_likelihood_rate": company["high_likelihood_rate"],
                "similarity_score": company["similarity_score"],
                "recent_summary": company["recent_summary"]
            }
            
            if include_summaries:
                row["recent_summary"] = company["recent_summary"]
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(rankings)} company rankings to {output_path}")
        return str(output_path) 