"""
Hiring Difficulties Analyzer

Analyzes 10-K filings for hiring difficulties, recruitment challenges, and workforce acquisition problems
using vector similarity search and AI-powered analysis instead of keyword matching.
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
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
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
class HiringDifficultiesAnalysis:
    """Data class for hiring difficulties analysis results"""
    cik: str
    company_name: str
    ticker: str
    year: int
    filing_url: str
    hiring_difficulty_score: float  # Vector similarity score (0-1)
    hiring_difficulty_likelihood: float  # AI-based likelihood score (0-10)
    hiring_difficulty_summary: str  # AI-generated summary
    top_similar_chunks: List[Dict]  # Top similar chunks found
    analyzed_date: str
    gvkey: Optional[str] = None
    error: Optional[str] = None


class HiringDifficultiesAnalyzer:
    """
    Analyzer for hiring difficulties and recruitment challenges in 10-K filings 
    using vector similarity search and AI-powered analysis
    """

    def __init__(self):
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=str(Path(settings.chroma_persist_directory) / "hiring_difficulties")
        )
        
        # Delay embedding model and LLM initialization to avoid fork issues
        self._embedding_model = None
        self._llm = None
        self._analysis_prompt = None
        self._similarity_prompt = None
        
        # Create collection for hiring difficulties reference patterns
        self.collection_name = "hiring_difficulties_patterns"
        self._reference_collection = None
        
        self.cache_file = Path(settings.cache_directory) / "hiring_difficulties_analysis.json"
        self.results: List[HiringDifficultiesAnalysis] = []
        
        # Initialize reference patterns for hiring difficulties
        self._initialize_reference_patterns()

    @property
    def embedding_model(self):
        """Lazy load embedding model to avoid fork issues"""
        if self._embedding_model is None:
            logger.info("Loading SentenceTransformer model for hiring difficulties analysis...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
        return self._embedding_model

    @property
    def llm(self):
        """Lazy load LLM to avoid fork issues"""
        if self._llm is None:
            logger.info("Initializing LLM for hiring difficulties analysis...")
            if settings.default_llm_provider == "openai":
                logger.info(f"Initializing OpenAI client with base_url: {settings.openai_base_url}")
                
                self._llm = ChatOpenAI(
                    model=settings.default_llm_model,
                    temperature=0.1,
                    max_tokens=1500,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    timeout=30,
                    max_retries=3
                )
            else:
                # Fallback to default OpenAI
                logger.info("Using fallback OpenAI client with default base_url")
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1500,
                    timeout=30,
                    max_retries=3
                )
            logger.info("LLM initialized successfully")
        return self._llm

    @property
    def analysis_prompt(self):
        """Lazy load analysis prompt to avoid fork issues"""
        if self._analysis_prompt is None:
            self._analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert analyst specializing in corporate hiring challenges and recruitment difficulties. 

Your task is to analyze document chunks from SEC filings and provide a comprehensive assessment of hiring difficulties based on the content and context.

Focus on:
1. Recruitment challenges and talent acquisition struggles
2. Difficulty finding qualified candidates or workers with desired skills
3. Employee retention issues and high turnover
4. Labor market constraints and competitive hiring environments
5. Staffing challenges and workforce availability issues
6. Geographic or industry-specific hiring difficulties
7. Cost pressures from recruitment and retention efforts
8. Impact on business operations due to hiring challenges

Provide a likelihood score from 0-10 where:
- 0-2: Very low likelihood of hiring difficulties
- 3-4: Low likelihood 
- 5-6: Moderate likelihood
- 7-8: High likelihood
- 9-10: Very high likelihood of significant hiring difficulties

Focus on the overall narrative and business context in the documents."""),
                ("human", """Analyze the following document chunks from {company_name} ({ticker}) for hiring difficulties and recruitment challenges.

Document chunks (ranked by relevance):
{document_context}

Based on this analysis, provide your response in the following JSON format:
{{
    "likelihood_score": <score_0_to_10>,
    "summary": "<detailed_summary_of_key_findings>",
    "key_indicators": ["<indicator1>", "<indicator2>", "<indicator3>"],
    "confidence": "<high/medium/low>",
    "business_impact": "<assessment_of_impact_on_business_operations>"
}}""")
            ])
        return self._analysis_prompt

    @property
    def similarity_prompt(self):
        """Lazy load similarity prompt for ranking companies"""
        if self._similarity_prompt is None:
            self._similarity_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert analyst specializing in comparative analysis of corporate hiring difficulties.

Your task is to analyze and compare hiring difficulty patterns across companies based on their document content and provide similarity scores."""),
                ("human", """Compare the hiring difficulty patterns in the following companies and provide similarity scores.

Companies and their hiring difficulty summaries:
{company_summaries}

Provide a ranking of companies from highest to lowest hiring difficulty likelihood, with brief explanations for each ranking.""")
            ])
        return self._similarity_prompt

    @property
    def reference_collection(self):
        """Lazy load reference collection"""
        if self._reference_collection is None:
            try:
                self._reference_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
            except Exception:
                # Create collection if it doesn't exist
                self._reference_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Reference patterns for hiring difficulties analysis"}
                )
        return self._reference_collection

    def _initialize_reference_patterns(self):
        """Initialize reference patterns for hiring difficulties in vector store"""
        hiring_difficulty_patterns = [
            "We are experiencing significant challenges in recruiting qualified candidates for our key positions, particularly in technical and specialized roles.",
            "The competitive labor market has made it increasingly difficult to attract and retain skilled workers, leading to longer recruitment cycles.",
            "High employee turnover rates have impacted our operational efficiency and increased our recruitment and training costs significantly.",
            "We face ongoing difficulties in filling critical positions due to a shortage of candidates with the required skills and experience.",
            "Labor market tightness has resulted in increased competition for talent and higher compensation costs to attract qualified employees.",
            "Our ability to expand operations has been constrained by challenges in hiring sufficient qualified personnel in our target markets.",
            "We have experienced delays in project execution due to difficulties in staffing key positions with appropriately skilled workers.",
            "The shortage of qualified candidates in our industry has led to extended recruitment processes and increased reliance on external contractors.",
            "Geographic constraints and limited local talent pools have made it challenging to staff our facilities in certain regions.",
            "Rising wage pressures and increased benefit costs reflect the competitive environment for attracting and retaining skilled employees.",
            "We have had to increase our recruitment budget and expand our search to national markets due to local talent shortages.",
            "High quit rates and difficulty replacing departing employees have impacted our ability to maintain optimal staffing levels.",
            "The specialized nature of our industry requires specific skills that are in short supply in the current labor market.",
            "We face challenges in hiring workers with the technical expertise required for our advanced manufacturing processes.",
            "Remote work preferences have limited our candidate pool for positions requiring on-site presence in our facilities."
        ]
        
        # Check if patterns are already loaded
        try:
            count = self.reference_collection.count()
            if count > 0:
                logger.info(f"Reference patterns already loaded: {count} patterns")
                return
        except Exception:
            pass
        
        # Load patterns into vector store
        logger.info("Loading hiring difficulty reference patterns into vector store...")
        
        for i, pattern in enumerate(hiring_difficulty_patterns):
            try:
                # Create embedding
                embedding = self.embedding_model.encode(pattern)
                
                # Add to collection
                self.reference_collection.add(
                    documents=[pattern],
                    embeddings=[embedding.tolist()],
                    ids=[f"pattern_{i:03d}"],
                    metadatas=[{"pattern_type": "hiring_difficulty", "pattern_id": i}]
                )
            except Exception as e:
                logger.warning(f"Error loading pattern {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(hiring_difficulty_patterns)} hiring difficulty reference patterns")

    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """
        Split text into overlapping chunks for better vector analysis
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                search_start = max(start + chunk_size - 200, start)
                sentence_ends = []

                for pattern in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    pos = text.rfind(pattern, search_start, end)
                    if pos != -1:
                        sentence_ends.append(pos + len(pattern))

                if sentence_ends:
                    end = max(sentence_ends)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start <= len(chunks[-1]) if chunks else 0:
                start = end

        return chunks

    def calculate_hiring_difficulty_similarity(self, text: str) -> Tuple[float, List[Dict]]:
        """
        Calculate hiring difficulty similarity using vector search against reference patterns
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (max_similarity_score, top_similar_chunks)
        """
        if not text or len(text.strip()) < 100:
            return 0.0, []

        try:
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Find most similar chunks to hiring difficulty patterns
            top_similar_chunks = []
            max_similarity = 0.0
            
            for i, chunk in enumerate(chunks):
                # Create embedding for chunk
                chunk_embedding = self.embedding_model.encode(chunk)
                
                # Search against reference patterns
                results = self.reference_collection.query(
                    query_embeddings=[chunk_embedding.tolist()],
                    n_results=3
                )
                
                if results['documents'] and results['documents'][0]:
                    # Get best similarity score (lower distance = higher similarity)
                    best_distance = min(results['distances'][0])
                    similarity_score = 1.0 / (1.0 + best_distance)  # Convert distance to similarity
                    
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                    
                    # Store chunk with similarity info
                    chunk_info = {
                        'chunk_index': i,
                        'content': chunk,
                        'similarity_score': similarity_score,
                        'best_distance': best_distance,
                        'matched_patterns': results['documents'][0][:2]  # Top 2 matched patterns
                    }
                    top_similar_chunks.append(chunk_info)
            
            # Sort chunks by similarity score and keep top ones
            top_similar_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_chunks = top_similar_chunks[:5]  # Keep top 5 most similar chunks
            
            logger.debug(f"Found {len(chunks)} chunks, max similarity: {max_similarity:.3f}")
            
            return max_similarity, top_chunks
            
        except Exception as e:
            logger.error(f"Error calculating hiring difficulty similarity: {e}")
            return 0.0, []

    def analyze_hiring_difficulties_with_ai(self, 
                                          text: str, 
                                          company_name: str, 
                                          ticker: str,
                                          top_chunks: List[Dict]) -> Tuple[float, str]:
        """
        Analyze hiring difficulties using AI with top similar chunks
        
        Args:
            text: Full text content
            company_name: Company name
            ticker: Company ticker
            top_chunks: Top similar chunks from vector search
            
        Returns:
            Tuple of (likelihood_score_0_to_10, summary_text)
        """
        if not top_chunks:
            return 0.0, "No relevant content found for hiring difficulties analysis"

        try:
            # Prepare document context from top similar chunks
            document_context = "\n\n".join([
                f"Chunk {i+1} (Similarity: {chunk['similarity_score']:.3f}):\n{chunk['content']}"
                for i, chunk in enumerate(top_chunks)
            ])
            
            # Limit context size
            if len(document_context) > 10000:
                document_context = document_context[:10000] + "\n...[content truncated]..."

            def ai_analysis():
                # Get AI analysis
                messages = self.analysis_prompt.format_messages(
                    company_name=company_name,
                    ticker=ticker,
                    document_context=document_context
                )
                response = log_llm_request(self.llm, messages, "for hiring difficulties analysis")
                return response

            # Use retry logic for AI analysis
            response = retry_with_backoff(ai_analysis, max_retries=3, base_delay=2.0)
            
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
                score_match = re.search(r'"likelihood_score":\s*(\d+(?:\.\d+)?)', response.content)
                if score_match:
                    likelihood_score = float(score_match.group(1))
                    likelihood_score = max(0.0, min(10.0, likelihood_score))
                    return likelihood_score, "AI analysis completed (partial parsing)"
                else:
                    logger.warning("Could not parse AI response, using fallback")
                    return 0.0, "AI analysis failed - parsing error"
                    
        except Exception as e:
            logger.error(f"Error in AI hiring difficulties analysis: {e}")
            return 0.0, f"AI analysis failed: {str(e)}"

    def analyze_edgar_filings(self, 
                            filings: List[EdgarFiling],
                            companies: List[SP500Company] = None) -> List[HiringDifficultiesAnalysis]:
        """
        Analyze EDGAR filings for hiring difficulties using vector similarity search
        
        Args:
            filings: List of EDGAR filings to analyze
            companies: Optional list of companies for additional metadata
            
        Returns:
            List of hiring difficulties analysis results
        """
        logger.info(f"Analyzing {len(filings)} EDGAR filings for hiring difficulties using vector similarity search")
        
        results = []
        company_lookup = {}
        
        # Create company lookup if provided
        if companies:
            company_lookup = {company.cik: company for company in companies}
        
        for i, filing in enumerate(filings):
            if i % 25 == 0:  # Log progress
                logger.info(f"Processed {i}/{len(filings)} filings")
                
            try:
                if not filing.text_content:
                    logger.warning(f"No text content for filing {filing.accession_number}")
                    continue
                
                # Calculate vector similarity score and get top similar chunks
                similarity_score, top_chunks = self.calculate_hiring_difficulty_similarity(filing.text_content)
                
                # Only proceed with AI analysis if there's some similarity
                if similarity_score > 0.1:  # Threshold for relevance
                    # Get AI-powered likelihood and summary
                    likelihood_score, summary = self.analyze_hiring_difficulties_with_ai(
                        filing.text_content, 
                        filing.company_name, 
                        company_lookup.get(filing.cik, {}).get('symbol', filing.cik) if company_lookup.get(filing.cik) else filing.cik,
                        top_chunks
                    )
                else:
                    likelihood_score = 0.0
                    summary = "No significant hiring difficulty patterns found"
                    top_chunks = []
                
                # Get company info
                company = company_lookup.get(filing.cik)
                ticker = company.symbol if company else filing.cik
                
                # Extract year from filing date
                year = int(filing.filing_date[:4]) if filing.filing_date else None
                
                analysis = HiringDifficultiesAnalysis(
                    cik=filing.cik,
                    company_name=filing.company_name,
                    ticker=ticker,
                    year=year,
                    filing_url=f"https://www.sec.gov/Archives/{filing.file_path}" if hasattr(filing, 'file_path') else "",
                    hiring_difficulty_score=similarity_score,
                    hiring_difficulty_likelihood=likelihood_score,
                    hiring_difficulty_summary=summary,
                    top_similar_chunks=[{
                        'chunk_index': chunk['chunk_index'],
                        'similarity_score': chunk['similarity_score'],
                        'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
                    } for chunk in top_chunks],
                    analyzed_date=pd.Timestamp.now().isoformat(),
                    gvkey=getattr(company, 'gvkey', None) if company else None
                )
                
                results.append(analysis)
                
                # Log significant findings
                if likelihood_score > 5.0 or similarity_score > 0.3:
                    logger.info(f"Found hiring difficulty likelihood {likelihood_score:.1f}/10 (similarity: {similarity_score:.3f}) for {ticker} ({year})")
                    
            except Exception as e:
                logger.error(f"Error analyzing filing {filing.accession_number}: {e}")
                continue
        
        self.results = results
        logger.info(f"Completed analysis. Found {len([r for r in results if r.hiring_difficulty_likelihood > 5.0])} filings with high hiring difficulty likelihood (>5.0)")
        
        return results

    def analyze_from_urls(self, 
                         df: pd.DataFrame,
                         limit: Optional[int] = None) -> List[HiringDifficultiesAnalysis]:
        """
        Analyze 10-K filings from URLs using vector similarity search
        
        Args:
            df: DataFrame with columns: cik, Year, FName, gvkey
            limit: Optional limit on number of filings to process
            
        Returns:
            List of hiring difficulties analysis results
        """
        logger.info(f"Analyzing filings from {len(df)} URLs with vector similarity search")
        
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
                    
                    # Calculate vector similarity and get top chunks
                    similarity_score, top_chunks = self.calculate_hiring_difficulty_similarity(text)
                    
                    # AI analysis if relevant
                    if similarity_score > 0.1:
                        likelihood_score, summary = self.analyze_hiring_difficulties_with_ai(
                            text, 
                            f"Company_{row['cik']}", 
                            str(row['cik']),
                            top_chunks
                        )
                    else:
                        likelihood_score = 0.0
                        summary = "No significant hiring difficulty patterns found"
                        top_chunks = []
                    
                    if likelihood_score > 3.0 or similarity_score > 0.2:
                        logger.info(f"Found hiring difficulty likelihood {likelihood_score:.1f}/10 (similarity: {similarity_score:.3f}) for CIK {row['cik']} ({row['Year']})")
                    
                    analysis = HiringDifficultiesAnalysis(
                        cik=str(row['cik']),
                        company_name=f"Company_{row['cik']}",
                        ticker=str(row['cik']),
                        year=int(row['Year']),
                        filing_url=url,
                        hiring_difficulty_score=similarity_score,
                        hiring_difficulty_likelihood=likelihood_score,
                        hiring_difficulty_summary=summary,
                        top_similar_chunks=[{
                            'chunk_index': chunk['chunk_index'],
                            'similarity_score': chunk['similarity_score'],
                            'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
                        } for chunk in top_chunks],
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
                                  min_similarity: float = 0.2,
                                  min_likelihood: float = 5.0) -> List[HiringDifficultiesAnalysis]:
        """
        Filter results to only significant hiring difficulty findings
        
        Args:
            min_similarity: Minimum vector similarity score
            min_likelihood: Minimum AI likelihood score (0-10)
            
        Returns:
            Filtered list of significant findings
        """
        return [
            result for result in self.results
            if result.hiring_difficulty_score >= min_similarity or result.hiring_difficulty_likelihood >= min_likelihood
        ]

    def get_company_hiring_difficulty_summary(self, cik: str) -> Dict:
        """
        Get hiring difficulty summary for a specific company
        
        Args:
            cik: Company CIK
            
        Returns:
            Summary dictionary
        """
        company_results = [r for r in self.results if r.cik == cik]
        
        if not company_results:
            return {"error": f"No results found for CIK {cik}"}
        
        avg_similarity_score = sum(r.hiring_difficulty_score for r in company_results) / len(company_results)
        avg_likelihood_score = sum(r.hiring_difficulty_likelihood for r in company_results) / len(company_results)
        max_likelihood_score = max(r.hiring_difficulty_likelihood for r in company_results)
        years_analyzed = sorted(list(set(r.year for r in company_results if r.year)))
        
        # Get the most recent summary
        recent_summary = ""
        if company_results:
            recent_result = max(company_results, key=lambda x: x.year if x.year else 0)
            recent_summary = recent_result.hiring_difficulty_summary
        
        return {
            "cik": cik,
            "company_name": company_results[0].company_name,
            "ticker": company_results[0].ticker,
            "average_similarity_score": avg_similarity_score,
            "average_likelihood_score": avg_likelihood_score,
            "max_likelihood_score": max_likelihood_score,
            "recent_summary": recent_summary,
            "years_analyzed": years_analyzed,
            "filings_count": len(company_results),
            "high_likelihood_filings": len([r for r in company_results if r.hiring_difficulty_likelihood > 6.0])
        }

    def rank_companies_by_hiring_difficulty(self, 
                                          min_filings: int = 1,
                                          recent_years_only: bool = False,
                                          years_lookback: int = 3) -> List[Dict]:
        """
        Rank all companies by hiring difficulties using vector similarity-based scoring
        
        Args:
            min_filings: Minimum number of filings required for a company to be included
            recent_years_only: If True, only consider filings from recent years
            years_lookback: Number of recent years to consider if recent_years_only is True
            
        Returns:
            List of dictionaries with company rankings, sorted from highest to lowest hiring difficulty
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
                    "total_similarity_score": 0,
                    "total_likelihood_score": 0,
                    "max_likelihood_score": 0,
                    "max_similarity_score": 0,
                    "years_analyzed": set(),
                    "high_likelihood_filings": 0,
                    "high_similarity_filings": 0,
                    "recent_summaries": []
                }
            
            company_data[company_key]["filings"].append(result)
            company_data[company_key]["total_similarity_score"] += result.hiring_difficulty_score
            company_data[company_key]["total_likelihood_score"] += result.hiring_difficulty_likelihood
            company_data[company_key]["max_likelihood_score"] = max(
                company_data[company_key]["max_likelihood_score"], 
                result.hiring_difficulty_likelihood
            )
            company_data[company_key]["max_similarity_score"] = max(
                company_data[company_key]["max_similarity_score"], 
                result.hiring_difficulty_score
            )
            
            if result.year:
                company_data[company_key]["years_analyzed"].add(result.year)
            
            if result.hiring_difficulty_likelihood > 6.0:
                company_data[company_key]["high_likelihood_filings"] += 1
            
            if result.hiring_difficulty_score > 0.3:
                company_data[company_key]["high_similarity_filings"] += 1
            
            # Collect recent summaries (non-empty ones)
            if result.hiring_difficulty_summary and result.hiring_difficulty_summary.strip() and len(result.hiring_difficulty_summary) > 20:
                company_data[company_key]["recent_summaries"].append({
                    "year": result.year,
                    "summary": result.hiring_difficulty_summary,
                    "likelihood": result.hiring_difficulty_likelihood,
                    "similarity": result.hiring_difficulty_score
                })
        
        # Filter companies by minimum filings requirement
        company_data = {k: v for k, v in company_data.items() if len(v["filings"]) >= min_filings}
        
        if not company_data:
            logger.warning("No companies meet the minimum filings requirement")
            return []
        
        # Calculate vector similarity-based ranking score for each company
        company_rankings = []
        for company_key, data in company_data.items():
            filings_count = len(data["filings"])
            avg_likelihood_score = data["total_likelihood_score"] / filings_count
            avg_similarity_score = data["total_similarity_score"] / filings_count
            high_likelihood_rate = data["high_likelihood_filings"] / filings_count
            high_similarity_rate = data["high_similarity_filings"] / filings_count
            
            # Get most recent and highest scoring summary
            recent_summary = ""
            if data["recent_summaries"]:
                # Sort by combined score (likelihood + similarity), then by year
                sorted_summaries = sorted(data["recent_summaries"], 
                                        key=lambda x: (x["likelihood"] + x["similarity"], x["year"] or 0), 
                                        reverse=True)
                recent_summary = sorted_summaries[0]["summary"]
            
            # Vector similarity-based scoring: combines AI likelihood with vector similarity
            similarity_score = (
                avg_likelihood_score * 0.4 +  # AI likelihood (semantic understanding)
                data["max_likelihood_score"] * 0.2 +  # Peak AI score
                avg_similarity_score * 10 * 0.2 +  # Vector similarity (scaled up)
                data["max_similarity_score"] * 10 * 0.1 +  # Peak vector similarity
                high_likelihood_rate * 2.0 +  # Consistency of high AI scores
                high_similarity_rate * 1.0  # Consistency of high vector similarity
            )
            
            company_ranking = {
                "rank": 0,  # Will be set after sorting
                "cik": data["cik"],
                "company_name": data["company_name"],
                "ticker": data["ticker"],
                "gvkey": data["gvkey"],
                "filings_analyzed": filings_count,
                "years_analyzed": sorted(list(data["years_analyzed"])),
                "avg_likelihood_score": round(avg_likelihood_score, 2),
                "max_likelihood_score": round(data["max_likelihood_score"], 2),
                "avg_similarity_score": round(avg_similarity_score, 3),
                "max_similarity_score": round(data["max_similarity_score"], 3),
                "high_likelihood_rate": round(high_likelihood_rate, 3),
                "high_similarity_rate": round(high_similarity_rate, 3),
                "similarity_score": round(similarity_score, 2),
                "recent_summary": recent_summary
            }
            
            company_rankings.append(company_ranking)
        
        # Sort by similarity score (highest first)
        company_rankings.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Assign ranks
        for i, company in enumerate(company_rankings):
            company["rank"] = i + 1
        
        logger.info(f"Ranked {len(company_rankings)} companies by hiring difficulty similarity")
        
        # Log top 10 companies
        top_10 = company_rankings[:10]
        logger.info("Top 10 companies with highest hiring difficulty similarity:")
        for company in top_10:
            logger.info(f"  {company['rank']}. {company['ticker']} - {company['company_name']} (Similarity Score: {company['similarity_score']})")
        
        return company_rankings

    def save_results(self, output_path: str = None) -> str:
        """
        Save analysis results to CSV file
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if not output_path:
            output_path = Path(settings.data_directory) / "hiring_difficulties_analysis.csv"
        
        # Convert results to DataFrame
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert top_similar_chunks to string for CSV
            result_dict['top_similar_chunks'] = json.dumps(result_dict['top_similar_chunks'])
            results_data.append(result_dict)
        
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
        Load results from cache file. If the cache is in an old format, it will be deleted.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check for outdated format by inspecting the first item
                if cached_data:
                    first_item = cached_data[0]
                    # If an old field is present or a new required field is missing, it's outdated.
                    if 'hiring_difficulty_mentions' in first_item or 'top_similar_chunks' not in first_item:
                         logger.warning("Outdated cache file format detected. Deleting cache file.")
                         self.cache_file.unlink()
                         return False

                self.results = [
                    HiringDifficultiesAnalysis(**result) for result in cached_data
                ]
                logger.info(f"Loaded {len(self.results)} results from cache")
                return True
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Could not load cache due to error. It might be corrupt or in an old format. Deleting cache file. Error: {e}")
            if self.cache_file.exists():
                self.cache_file.unlink()
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading cache: {e}")
        
        return False

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the analysis including vector similarity scores
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No analysis results available"}
        
        total_filings = len(self.results)
        filings_with_high_similarity = len([r for r in self.results if r.hiring_difficulty_score > 0.3])
        filings_with_high_likelihood = len([r for r in self.results if r.hiring_difficulty_likelihood > 6.0])
        
        # Calculate average scores
        avg_similarity = sum(r.hiring_difficulty_score for r in self.results) / total_filings if total_filings > 0 else 0
        avg_likelihood = sum(r.hiring_difficulty_likelihood for r in self.results) / total_filings if total_filings > 0 else 0
        
        # Company and year breakdowns
        companies = set(r.cik for r in self.results)
        years = set(r.year for r in self.results if r.year)
        
        # Top companies by likelihood score
        company_likelihood = {}
        for result in self.results:
            key = f"{result.ticker} ({result.cik})"
            if key not in company_likelihood:
                company_likelihood[key] = []
            company_likelihood[key].append(result.hiring_difficulty_likelihood)
        
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
        
        # Top companies by similarity score
        company_similarity = {}
        for result in self.results:
            key = f"{result.ticker} ({result.cik})"
            if key not in company_similarity:
                company_similarity[key] = []
            company_similarity[key].append(result.hiring_difficulty_score)
        
        company_avg_similarity = {
            company: sum(scores) / len(scores) 
            for company, scores in company_similarity.items()
        }
        
        top_companies_by_similarity = sorted(
            company_avg_similarity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_filings_analyzed": total_filings,
            "filings_with_high_similarity": filings_with_high_similarity,
            "filings_with_high_likelihood": filings_with_high_likelihood,
            "high_similarity_rate": filings_with_high_similarity / total_filings if total_filings > 0 else 0,
            "high_likelihood_rate": filings_with_high_likelihood / total_filings if total_filings > 0 else 0,
            "average_similarity_score": avg_similarity,
            "average_likelihood_score": avg_likelihood,
            "unique_companies": len(companies),
            "years_covered": sorted(list(years)),
            "top_companies_by_likelihood": top_companies_by_likelihood,
            "top_companies_by_similarity": top_companies_by_similarity,
            "analysis_method": "vector_similarity_search"
        }

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
            rankings = self.rank_companies_by_hiring_difficulty()
        
        if not rankings:
            raise ValueError("No rankings available to export")
        
        if not output_path:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(settings.data_directory) / f"hiring_difficulties_company_rankings_{timestamp}.csv"
        
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
                "avg_similarity_score": company["avg_similarity_score"],
                "max_similarity_score": company["max_similarity_score"],
                "high_likelihood_rate": company["high_likelihood_rate"],
                "high_similarity_rate": company["high_similarity_rate"],
                "similarity_score": company["similarity_score"]
            }
            
            if include_summaries:
                row["recent_summary"] = company["recent_summary"]
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(rankings)} company rankings to {output_path}")
        return str(output_path) 