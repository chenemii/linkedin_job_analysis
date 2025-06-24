"""
Simplified Hiring Difficulties Analyzer

Analyzes 10-K filings for hiring difficulties by summarizing relevant sections using an LLM.
"""

import logging
import os
import json
import re
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..config import settings
from .edgar_collector import EdgarFiling

logger = logging.getLogger(__name__)

# Fix tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class HiringDifficultySummary:
    """Data class for hiring difficulties analysis results."""
    cik: str
    company_name: str
    ticker: str
    year: int
    filing_url: str
    is_mentioned: bool
    summary: str
    chunks_analyzed: Optional[str] = None  # The actual text chunks that were analyzed
    error: Optional[str] = None

class HiringDifficultiesAnalyzer:
    """
    Analyzes 10-K filings for hiring difficulties using an LLM to summarize relevant content.
    """

    def __init__(self):
        """Initializes the analyzer."""
        self._llm = None
        self._analysis_prompt = None
        self.results: List[HiringDifficultySummary] = []
        
        # Keywords for hiring difficulties (for similarity ranking)
        self.hiring_keywords = [
            'hiring', 'recruitment', 'talent acquisition', 'labor shortage', 'workforce',
            'staffing', 'employees', 'human resources', 'retain talent', 'turnover',
            'skill shortage', 'qualified candidates', 'competitive market', 'labor market',
            'attract talent', 'recruiting', 'personnel', 'human capital', 'shortage of workers',
            'difficulty finding', 'challenge recruiting', 'talent pool', 'labor scarcity',
            'retention', 'attrition', 'employee shortage', 'skilled workers', 'labor constraints'
        ]
        
        # Compile regex patterns for faster matching
        self.keyword_patterns = [
            re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in self.hiring_keywords
        ]

    def _calculate_chunk_similarity(self, chunk: str) -> float:
        """
        Calculate similarity score for a chunk based on hiring difficulty keywords.
        
        Args:
            chunk: Text chunk to score
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not chunk:
            return 0.0
        
        chunk_lower = chunk.lower()
        keyword_matches = 0
        total_matches = 0
        
        for pattern in self.keyword_patterns:
            matches = len(pattern.findall(chunk))
            if matches > 0:
                keyword_matches += 1
                total_matches += matches
        
        # Calculate score based on:
        # - Number of different keywords found (diversity)
        # - Total number of keyword matches (frequency)
        # - Relative to chunk length (density)
        
        diversity_score = keyword_matches / len(self.hiring_keywords)
        frequency_score = min(total_matches / 10, 1.0)  # Cap at 10 matches
        density_score = min(total_matches / (len(chunk) / 1000), 1.0)  # Matches per 1000 chars
        
        # Weighted combination
        final_score = (diversity_score * 0.4 + frequency_score * 0.4 + density_score * 0.2)
        
        return final_score

    def _rank_chunks_by_similarity(self, chunks: List[str]) -> List[Tuple[str, float]]:
        """
        Rank chunks by their similarity to hiring difficulties content.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of (chunk, score) tuples sorted by score (highest first)
        """
        scored_chunks = []
        
        for chunk in chunks:
            score = self._calculate_chunk_similarity(chunk)
            scored_chunks.append((chunk, score))
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy load LLM to avoid fork issues in multiprocessing environments."""
        if self._llm is None:
            logger.info("Initializing LLM for hiring difficulties analysis...")
            self._llm = ChatOpenAI(
                model=settings.default_llm_model,
                temperature=0.0,
                max_tokens=2048,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=60,
                max_retries=2,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            logger.info("LLM initialized successfully.")
        return self._llm

    @property
    def analysis_prompt(self) -> ChatPromptTemplate:
        """Lazy load analysis prompt."""
        if self._analysis_prompt is None:
            self._analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert financial analyst. Your task is to analyze excerpts from a company's 10-K filing.
Focus exclusively on identifying and summarizing any statements related to hiring difficulties, recruitment challenges, labor shortages, or problems attracting and retaining talent.

If the text mentions any of these issues, provide a concise summary of the key points.
If there are no mentions of hiring difficulties, you must state that clearly.
                 
Quote from the text to explain your answer.                

Respond in JSON format with two keys: "is_mentioned" (a boolean true/false) and "summary" (your summary as a string, or a statement that no difficulties were mentioned)."""),
                ("human", "Please analyze the following text from a 10-K filing:\n\n---\n\n{document_text}\n\n---")
            ])
        return self._analysis_prompt

    def _chunk_text(self, text: str, chunk_size: int = 25000, overlap: int = 500) -> List[str]:
        """Splits text into overlapping chunks."""
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    def _analyze_text(self, text: str) -> Tuple[bool, str]:
        """
        Analyzes a single block of text for hiring difficulties using an LLM.
        This is a helper for `analyze_filing_text`.
        """
        messages = self.analysis_prompt.format_messages(document_text=text)
        response = self.llm.invoke(messages)
        result = json.loads(response.content)
        return result.get("is_mentioned", False), result.get("summary", "")

    def analyze_filing_text(self, text: str) -> Tuple[bool, str, str]:
        """
        Analyzes a single filing's full text for hiring difficulties by chunking and ranking.
        Combines relevant chunks and performs a single analysis for efficiency.
        
        Returns:
            Tuple of (is_mentioned, summary, chunks_analyzed)
        """
        if not text or len(text.strip()) < 100:
            return False, "Document text is too short to analyze.", ""

        chunks = self._chunk_text(text)
        
        # Rank chunks by similarity to hiring difficulties content
        ranked_chunks = self._rank_chunks_by_similarity(chunks)
        
        # Log ranking results
        top_scores = [score for _, score in ranked_chunks[:5]]
        logger.info(f"Chunk ranking complete. Total chunks: {len(chunks)}, Top 5 scores: {top_scores}")
        
        # Limit the number of chunks to analyze, prioritizing high-scoring ones
        max_chunks_to_analyze = min(3, len(ranked_chunks))
        
        # Filter chunks with score > 0 (at least some keywords) and take top ones
        relevant_chunks = [(chunk, score) for chunk, score in ranked_chunks if score > 0]
        
        if relevant_chunks:
            # Take top scoring chunks
            selected_chunks = relevant_chunks[:max_chunks_to_analyze]
            logger.info(f"Selected top {len(selected_chunks)} relevant chunks (scores: {[f'{s:.3f}' for _, s in selected_chunks]})")
            
            # Combine all selected chunks into one text block for analysis
            combined_text = "\n\n--- SECTION ---\n\n".join([chunk for chunk, _ in selected_chunks])
            
            # Create a formatted version for reference with scores
            chunks_reference = ""
            for i, (chunk, score) in enumerate(selected_chunks, 1):
                chunks_reference += f"\n\n=== CHUNK {i} (Relevance Score: {score:.3f}) ===\n"
                chunks_reference += chunk[:2000] + ("..." if len(chunk) > 2000 else "")
                chunks_reference += f"\n(Total length: {len(chunk):,} characters)"
            
        else:
            # If no chunks have keywords, take first few chunks as fallback
            fallback_chunks = ranked_chunks[:min(2, max_chunks_to_analyze)]
            logger.info(f"No keyword matches found. Using first {len(fallback_chunks)} chunks as fallback.")
            
            # Combine fallback chunks for analysis
            combined_text = "\n\n--- SECTION ---\n\n".join([chunk for chunk, _ in fallback_chunks])
            
            # Create reference for fallback chunks
            chunks_reference = ""
            for i, (chunk, score) in enumerate(fallback_chunks, 1):
                chunks_reference += f"\n\n=== FALLBACK CHUNK {i} (Score: {score:.3f}) ===\n"
                chunks_reference += chunk[:2000] + ("..." if len(chunk) > 2000 else "")
                chunks_reference += f"\n(Total length: {len(chunk):,} characters)"

        # Perform single analysis on combined text
        try:
            logger.info(f"Performing single analysis on combined text ({len(combined_text):,} characters)")
            is_mentioned, summary = self._analyze_text(combined_text)
            
            if not is_mentioned:
                return False, "No hiring difficulties mentioned in the analyzed sections.", chunks_reference
            
            # Enhance summary with source reference
            enhanced_summary = f"{summary}\n\n--- SOURCE TEXT ANALYZED ---{chunks_reference}"
            
            return True, enhanced_summary, chunks_reference
            
        except Exception as e:
            logger.error(f"Error analyzing combined text: {e}")
            return False, f"Analysis failed: {str(e)}", chunks_reference

    def analyze_edgar_filings(self, filings: List[EdgarFiling], companies: Optional[List['SP500Company']] = None) -> List[HiringDifficultySummary]:
        """
        Analyzes a list of EDGAR filings for hiring difficulties.
        """
        logger.info(f"Starting analysis of {len(filings)} EDGAR filings.")
        # Use a local list for the results of this run
        local_results = []
        company_lookup = {company.cik: company for company in companies} if companies else {}
        
        for i, filing in enumerate(filings):
            logger.info(f"Processing {i+1}/{len(filings)}: {filing.company_name} ({filing.cik})")
            summary = None
            error_message = None
            is_mentioned = False
            
            try:
                if not filing.text_content:
                    error_message = "No text content available for filing."
                else:
                    is_mentioned, summary, chunks_analyzed = self.analyze_filing_text(filing.text_content)
            
            except Exception as e:
                logger.error(f"Error analyzing filing {filing.accession_number}: {e}", exc_info=True)
                error_message = str(e)

            year = int(filing.filing_date[:4]) if filing.filing_date else None
            
            company_info = company_lookup.get(filing.cik)
            ticker = company_info.symbol if company_info else filing.cik
            
            # A more robust URL to the filing's index page
            accession_no_dashes = filing.accession_number.replace('-', '')
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{filing.cik}/{accession_no_dashes}/{filing.accession_number}-index.html"

            local_results.append(HiringDifficultySummary(
                cik=filing.cik,
                company_name=filing.company_name,
                ticker=ticker,
                year=year,
                filing_url=filing_url,
                is_mentioned=is_mentioned,
                summary=summary or "Analysis could not be completed.",
                chunks_analyzed=chunks_analyzed,
                error=error_message,
            ))
        
        # Store the results of this run in the instance variable for saving
        self.results = local_results
        logger.info(f"Completed analysis of {len(filings)} filings.")
        return self.results

    def save_results(self, output_path: str = None) -> str:
        """Saves analysis results to a CSV file."""
        if not self.results:
            logger.warning("No results to save.")
            return ""

        if not output_path:
            output_dir = Path(settings.data_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "hiring_difficulties_summary.csv"
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(self.results)} results to {output_path}")
        return str(output_path) 