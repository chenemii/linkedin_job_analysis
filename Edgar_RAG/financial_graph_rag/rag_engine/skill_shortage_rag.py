"""
Skill Shortage RAG Engine

Specialized RAG engine for analyzing skill shortages and talent gaps
in financial documents with AI-based likelihood scoring.
"""

import logging
import time
import random
import json
from typing import List, Dict, Optional
from pathlib import Path

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .vector_rag import VectorRAGEngine, retry_with_backoff, log_llm_request
from ..config import settings
from ..data_collectors.skill_shortage_analyzer import SkillShortageAnalyzer, SkillShortageAnalysis

logger = logging.getLogger(__name__)


class SkillShortageRAGEngine(VectorRAGEngine):
    """
    Specialized RAG engine for skill shortage analysis
    Extends VectorRAGEngine with skill shortage specific functionality
    """

    def __init__(self):
        """Initialize the Skill Shortage RAG engine"""
        super().__init__()
        
        # Initialize skill shortage analyzer
        self.skill_shortage_analyzer = SkillShortageAnalyzer()
        
        # Create skill shortage specific prompts
        self.skill_shortage_analysis_prompt = self._create_skill_shortage_analysis_prompt()
        self.skill_shortage_comparison_prompt = self._create_skill_shortage_comparison_prompt()

    def _create_skill_shortage_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for skill shortage analysis"""
        system_prompt = """You are a specialized analyst focusing on workforce challenges and skill shortages in corporate environments.
        
        You analyze SEC filings and financial documents to identify and interpret mentions of:
        - Skills gaps and talent shortages
        - Recruitment and retention challenges  
        - Workforce development needs
        - Competition for skilled talent
        - Training and development initiatives
        - Impact of skill shortages on business operations
        
        Provide detailed, actionable insights based on the document evidence provided."""

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Based on the following documents, analyze skill shortage and talent gap issues for the company: {query}

            Document Context:
            {document_context}

            Skill Shortage Analysis Results:
            {skill_shortage_data}

            Please provide a comprehensive analysis covering:
            
            1. **Skill Shortage Overview**
               - Types of skills in short supply
               - Severity and frequency of mentions
               - Trends over time
            
            2. **Business Impact**
               - How skill shortages affect operations
               - Financial implications mentioned
               - Strategic challenges created
            
            3. **Company Response**
               - Recruitment strategies mentioned
               - Training and development programs
               - Retention initiatives
               - Partnerships with educational institutions
            
            4. **Industry Context**
               - How this compares to industry trends
               - Competitive implications
               - Market dynamics affecting talent supply
            
            5. **Recommendations**
               - Strategic recommendations for addressing skill gaps
               - Areas requiring immediate attention
               - Long-term workforce development strategies
            
            Provide specific evidence from the documents to support your analysis.
            """)
        ])

        return analysis_prompt

    def _create_skill_shortage_comparison_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for comparing skill shortage across companies"""
        system_prompt = """You are an expert analyst specializing in comparative workforce analysis across companies and industries.
        
        You analyze skill shortage patterns, trends, and strategies across multiple organizations to provide insights about:
        - Industry-wide talent challenges
        - Best practices in addressing skill gaps
        - Competitive dynamics in talent acquisition
        - Sector-specific workforce trends"""

        comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Based on the skill shortage analysis data from multiple companies, provide a comparative analysis: {query}

            Skill Shortage Data:
            {skill_shortage_comparison_data}

            Document Context (if available):
            {document_context}

            Please provide analysis covering:
            
            1. **Cross-Company Comparison**
               - Companies most affected by skill shortages
               - Variation in skill shortage types and severity
               - Industry patterns and trends
            
            2. **Sector Analysis**
               - Which sectors face the greatest challenges
               - Sector-specific skill gaps
               - Regional or market factors
            
            3. **Strategic Patterns**
               - Common approaches to addressing skill shortages
               - Innovative solutions identified
               - Success factors and best practices
            
            4. **Market Implications**
               - Competitive advantages/disadvantages
               - Investment implications
               - Long-term industry outlook
            
            5. **Recommendations**
               - Industry-wide recommendations
               - Company-specific insights
               - Policy or systemic changes needed
            """)
        ])

        return comparison_prompt

    def analyze_company_skill_shortage(self, 
                                     company_ticker: str,
                                     analysis_focus: str = "comprehensive skill shortage analysis") -> Dict:
        """
        Analyze skill shortage issues for a specific company
        
        Args:
            company_ticker: Company ticker symbol
            analysis_focus: Specific aspect to analyze
            
        Returns:
            Comprehensive skill shortage analysis results with document chunk references
        """
        logger.info(f"Analyzing skill shortage for {company_ticker}")

        # Get relevant documents using parent class functionality
        base_analysis = self.analyze_company_ma_impact(
            query=f"skill shortage, talent gap, recruitment challenges, workforce development for {company_ticker}",
            company_ticker=company_ticker
        )

        if 'error' in base_analysis:
            return base_analysis

        # Get skill shortage specific data if available
        skill_shortage_data = self._get_company_skill_shortage_data(company_ticker)

        # Combine document context with skill shortage analysis
        document_context = base_analysis.get('document_context', '')
        
        # Get the retrieved chunks for source references
        retrieved_chunks = base_analysis.get('retrieved_chunks', [])
        
        # Create analysis query
        query = f"Analyze {analysis_focus} for {company_ticker}"

        # Generate skill shortage specific analysis with retry logic
        try:
            messages = self.skill_shortage_analysis_prompt.format_messages(
                query=query,
                document_context=document_context,
                skill_shortage_data=json.dumps(skill_shortage_data, indent=2)
            )

            def llm_call():
                return log_llm_request(self.llm, messages, f"for {company_ticker} skill shortage analysis")
            
            # Use retry logic for LLM call
            response = retry_with_backoff(llm_call, max_retries=3, base_delay=2.0)
            skill_shortage_analysis = response.content

            # Prepare source document references
            source_documents = []
            for i, chunk in enumerate(retrieved_chunks):
                source_doc = {
                    'chunk_id': chunk['id'],
                    'chunk_index': i + 1,
                    'similarity_score': chunk['similarity_score'],
                    'distance': chunk['distance'],
                    'content': chunk.get('content', ''),
                    'metadata': chunk['metadata'],
                    'chunk_size': chunk['chunk_size'],
                    'original_document_id': chunk['metadata'].get('original_document_id', 'Unknown'),
                    'filing_date': chunk['metadata'].get('filing_date', 'Unknown'),
                    'form_type': chunk['metadata'].get('form_type', 'Unknown'),
                    'company_name': chunk['metadata'].get('company_name', company_ticker)
                }
                source_documents.append(source_doc)

            # Create document reference summary
            document_references = {
                'total_chunks_analyzed': len(retrieved_chunks),
                'chunks_used_in_analysis': len(retrieved_chunks),
                'source_documents': source_documents,
                'document_summary': self._create_document_reference_summary(retrieved_chunks)
            }

            # Combine results
            results = {
                **base_analysis,
                'skill_shortage_analysis': skill_shortage_analysis,
                'skill_shortage_data': skill_shortage_data,
                'analysis_type': 'skill_shortage',
                'document_references': document_references,
                'source_documents': source_documents  # For backward compatibility
            }

            return results

        except Exception as e:
            logger.error(f"Error in skill shortage analysis: {e}")
            return {
                **base_analysis,
                'skill_shortage_error': str(e),
                'skill_shortage_data': skill_shortage_data,
                'document_references': {
                    'total_chunks_analyzed': len(retrieved_chunks),
                    'source_documents': retrieved_chunks,
                    'error': f"Analysis failed: {str(e)}"
                }
            }

    def _create_document_reference_summary(self, chunks: List[Dict]) -> Dict:
        """
        Create a summary of document references used in analysis
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Summary of document references
        """
        if not chunks:
            return {'error': 'No document chunks available'}

        # Group by original document
        documents_by_id = {}
        for chunk in chunks:
            doc_id = chunk['metadata'].get('original_document_id', 'Unknown')
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    'document_id': doc_id,
                    'chunks': [],
                    'total_chunks': 0,
                    'filing_date': chunk['metadata'].get('filing_date', 'Unknown'),
                    'form_type': chunk['metadata'].get('form_type', 'Unknown'),
                    'company_name': chunk['metadata'].get('company_name', 'Unknown'),
                    'avg_similarity': 0
                }
            
            documents_by_id[doc_id]['chunks'].append({
                'chunk_id': chunk['id'],
                'chunk_index': chunk['metadata'].get('chunk_index', 0),
                'similarity_score': chunk['similarity_score'],
                'content_preview': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
            })
            documents_by_id[doc_id]['total_chunks'] += 1

        # Calculate average similarity for each document
        for doc_id, doc_info in documents_by_id.items():
            if doc_info['chunks']:
                doc_info['avg_similarity'] = sum(chunk['similarity_score'] for chunk in doc_info['chunks']) / len(doc_info['chunks'])

        # Sort documents by average similarity
        sorted_documents = sorted(documents_by_id.values(), key=lambda x: x['avg_similarity'], reverse=True)

        return {
            'unique_documents': len(documents_by_id),
            'total_chunks': len(chunks),
            'documents': sorted_documents,
            'highest_similarity_chunk': max(chunks, key=lambda x: x['similarity_score']) if chunks else None,
            'date_range': {
                'earliest': min([chunk['metadata'].get('filing_date', '9999-12-31') for chunk in chunks if chunk['metadata'].get('filing_date')], default='Unknown'),
                'latest': max([chunk['metadata'].get('filing_date', '0000-01-01') for chunk in chunks if chunk['metadata'].get('filing_date')], default='Unknown')
            }
        }

    def compare_skill_shortage_across_companies(self, 
                                              company_tickers: List[str] = None,
                                              sector: str = None,
                                              analysis_focus: str = "comparative skill shortage analysis") -> Dict:
        """
        Compare skill shortage patterns across multiple companies
        
        Args:
            company_tickers: Optional list of specific companies to compare
            sector: Optional sector filter
            analysis_focus: Specific aspect to analyze
            
        Returns:
            Comparative skill shortage analysis results
        """
        logger.info("Performing comparative skill shortage analysis")

        # Get skill shortage data for comparison
        if company_tickers:
            comparison_data = {}
            for ticker in company_tickers:
                comparison_data[ticker] = self._get_company_skill_shortage_data(ticker)
        else:
            # Get all available skill shortage data
            comparison_data = self._get_all_skill_shortage_data(sector=sector)

        if not comparison_data:
            return {"error": "No skill shortage data available for comparison"}

        # Get relevant documents for context
        document_context = ""
        if company_tickers and len(company_tickers) <= 5:  # Limit to avoid too much context
            for ticker in company_tickers[:5]:
                try:
                    docs = self.retrieve_company_documents(
                        query="skill shortage talent gap recruitment workforce",
                        company_ticker=ticker,
                        top_k=2
                    )
                    for doc in docs:
                        document_context += f"\n--- {ticker} ---\n{doc['content'][:500]}...\n"
                except:
                    continue

        # Create comparison query with retry logic
        query = f"Compare and analyze {analysis_focus} across companies"

        try:
            messages = self.skill_shortage_comparison_prompt.format_messages(
                query=query,
                skill_shortage_comparison_data=json.dumps(comparison_data, indent=2),
                document_context=document_context
            )

            def llm_call():
                return log_llm_request(self.llm, messages, f"for comparative skill shortage analysis")
            
            # Use retry logic for LLM call
            response = retry_with_backoff(llm_call, max_retries=3, base_delay=2.0)
            comparative_analysis = response.content

            return {
                'comparative_analysis': comparative_analysis,
                'skill_shortage_comparison_data': comparison_data,
                'companies_analyzed': list(comparison_data.keys()),
                'analysis_type': 'skill_shortage_comparison'
            }

        except Exception as e:
            logger.error(f"Error in comparative skill shortage analysis: {e}")
            return {
                'error': str(e),
                'skill_shortage_comparison_data': comparison_data
            }

    def analyze_skill_shortage_trends(self, 
                                    years: List[int] = None,
                                    sector: str = None) -> Dict:
        """
        Analyze skill shortage trends over time and across sectors
        
        Args:
            years: Optional list of years to analyze
            sector: Optional sector filter
            
        Returns:
            Trend analysis results
        """
        logger.info("Analyzing skill shortage trends")

        # Load skill shortage analysis results
        if not self.skill_shortage_analyzer.load_cache():
            return {"error": "No skill shortage analysis data available. Run skill shortage analysis first."}

        # Filter results based on parameters
        results = self.skill_shortage_analyzer.results
        if years:
            results = [r for r in results if r.year in years]
        if sector:
            # Would need company sector information for filtering
            pass

        if not results:
            return {"error": "No data available for specified criteria"}

        # Generate trend analysis
        trend_data = self._calculate_skill_shortage_trends(results)
        
        return {
            'trend_analysis': trend_data,
            'analysis_type': 'skill_shortage_trends',
            'data_points': len(results),
            'years_covered': sorted(list(set(r.year for r in results if r.year))),
            'companies_covered': len(set(r.cik for r in results))
        }

    def _get_company_skill_shortage_data(self, company_ticker: str) -> Dict:
        """Get skill shortage data for a specific company including AI likelihood scores"""
        # Try to load from cache first
        if not hasattr(self.skill_shortage_analyzer, 'results') or not self.skill_shortage_analyzer.results:
            self.skill_shortage_analyzer.load_cache()

        # Find company data (would need CIK lookup in real implementation)
        company_results = []
        for result in self.skill_shortage_analyzer.results:
            if result.ticker == company_ticker or result.cik == company_ticker:
                company_results.append(result)

        if not company_results:
            return {"error": f"No skill shortage data found for {company_ticker}"}

        # Aggregate data
        total_mentions = sum(r.skill_shortage_mentions for r in company_results)
        avg_keyword_score = sum(r.skill_shortage_score for r in company_results) / len(company_results)
        avg_likelihood_score = sum(r.skill_shortage_likelihood for r in company_results) / len(company_results)
        years_with_data = sorted(list(set(r.year for r in company_results if r.year)))
        
        # Get the most recent summary
        recent_summary = ""
        if company_results:
            recent_result = max(company_results, key=lambda x: x.year if x.year else 0)
            recent_summary = recent_result.skill_shortage_summary

        return {
            "company_ticker": company_ticker,
            "total_mentions": total_mentions,
            "average_keyword_score": avg_keyword_score,
            "average_likelihood_score": avg_likelihood_score,
            "recent_summary": recent_summary,
            "filings_analyzed": len(company_results),
            "years_with_data": years_with_data,
            "filings_with_mentions": len([r for r in company_results if r.skill_shortage_mentions > 0]),
            "filings_with_high_likelihood": len([r for r in company_results if r.skill_shortage_likelihood > 6.0]),
            "detailed_results": [
                {
                    "year": r.year,
                    "mentions": r.skill_shortage_mentions,
                    "keyword_score": r.skill_shortage_score,
                    "likelihood_score": r.skill_shortage_likelihood,
                    "summary": r.skill_shortage_summary,
                    "filing_url": r.filing_url
                } for r in company_results
            ]
        }

    def _get_all_skill_shortage_data(self, sector: str = None) -> Dict:
        """Get skill shortage data for all companies including AI likelihood scores"""
        if not hasattr(self.skill_shortage_analyzer, 'results') or not self.skill_shortage_analyzer.results:
            self.skill_shortage_analyzer.load_cache()

        if not self.skill_shortage_analyzer.results:
            return {}

        # Group by company
        company_data = {}
        for result in self.skill_shortage_analyzer.results:
            ticker = result.ticker
            if ticker not in company_data:
                company_data[ticker] = []
            company_data[ticker].append(result)

        # Aggregate data for each company
        aggregated_data = {}
        for ticker, results in company_data.items():
            total_mentions = sum(r.skill_shortage_mentions for r in results)
            avg_likelihood = sum(r.skill_shortage_likelihood for r in results) / len(results)
            
            # Include companies with either keyword mentions OR high likelihood scores
            if total_mentions > 0 or avg_likelihood > 5.0:
                aggregated_data[ticker] = {
                    "total_mentions": total_mentions,
                    "average_keyword_score": sum(r.skill_shortage_score for r in results) / len(results),
                    "average_likelihood_score": avg_likelihood,
                    "filings_count": len(results),
                    "filings_with_high_likelihood": len([r for r in results if r.skill_shortage_likelihood > 6.0]),
                    "years_covered": sorted(list(set(r.year for r in results if r.year)))
                }

        return aggregated_data

    def _calculate_skill_shortage_trends(self, results: List[SkillShortageAnalysis]) -> Dict:
        """Calculate trend statistics from skill shortage results including AI likelihood scores"""
        # Group by year
        year_data = {}
        for result in results:
            if not result.year:
                continue
            if result.year not in year_data:
                year_data[result.year] = {
                    "total_mentions": 0,
                    "total_likelihood_score": 0,
                    "filings_count": 0,
                    "companies": set(),
                    "filings_with_mentions": 0,
                    "filings_with_high_likelihood": 0
                }
            
            year_data[result.year]["total_mentions"] += result.skill_shortage_mentions
            year_data[result.year]["total_likelihood_score"] += result.skill_shortage_likelihood
            year_data[result.year]["filings_count"] += 1
            year_data[result.year]["companies"].add(result.cik)
            
            if result.skill_shortage_mentions > 0:
                year_data[result.year]["filings_with_mentions"] += 1
            
            if result.skill_shortage_likelihood > 6.0:
                year_data[result.year]["filings_with_high_likelihood"] += 1

        # Calculate trend metrics
        trend_metrics = {}
        for year, data in year_data.items():
            trend_metrics[year] = {
                "total_mentions": data["total_mentions"],
                "avg_likelihood_score": data["total_likelihood_score"] / data["filings_count"] if data["filings_count"] > 0 else 0,
                "filings_analyzed": data["filings_count"],
                "companies_count": len(data["companies"]),
                "mention_rate": data["filings_with_mentions"] / data["filings_count"] if data["filings_count"] > 0 else 0,
                "high_likelihood_rate": data["filings_with_high_likelihood"] / data["filings_count"] if data["filings_count"] > 0 else 0,
                "avg_mentions_per_filing": data["total_mentions"] / data["filings_count"] if data["filings_count"] > 0 else 0
            }

        return {
            "yearly_trends": trend_metrics,
            "overall_stats": {
                "total_years": len(year_data),
                "total_filings": sum(data["filings_count"] for data in year_data.values()),
                "total_mentions": sum(data["total_mentions"] for data in year_data.values()),
                "avg_likelihood_score": sum(data["total_likelihood_score"] for data in year_data.values()) / sum(data["filings_count"] for data in year_data.values()) if sum(data["filings_count"] for data in year_data.values()) > 0 else 0,
                "unique_companies": len(set().union(*[data["companies"] for data in year_data.values()]))
            }
        }

    def add_skill_shortage_analysis_to_vector_store(self, 
                                                   skill_shortage_results: List[SkillShortageAnalysis]):
        """
        Add skill shortage analysis results to the vector store for enhanced retrieval
        
        Args:
            skill_shortage_results: List of skill shortage analysis results
        """
        logger.info(f"Adding {len(skill_shortage_results)} skill shortage analyses to vector store")

        for result in skill_shortage_results:
            # Include results with either keyword mentions OR high likelihood scores
            if result.skill_shortage_mentions > 0 or result.skill_shortage_likelihood > 5.0:
                # Create document text summarizing the skill shortage findings
                doc_text = f"""
                Skill Shortage Analysis for {result.company_name} ({result.ticker}) - {result.year}
                
                AI Likelihood Score: {result.skill_shortage_likelihood:.1f}/10
                Keyword Mentions: {result.skill_shortage_mentions}
                Keyword Score: {result.skill_shortage_score:.4f}
                
                AI Summary: {result.skill_shortage_summary}
                
                This filing shows a {result.skill_shortage_likelihood:.1f}/10 likelihood of skill shortages based on AI analysis
                of language patterns and business context, with {result.skill_shortage_mentions} explicit keyword mentions.
                
                Filing URL: {result.filing_url}
                Analysis Date: {result.analyzed_date}
                """

                # Add to company-specific vector store
                try:
                    self.add_document_to_company_store(
                        document_text=doc_text,
                        document_id=f"skill_shortage_{result.cik}_{result.year}",
                        company_ticker=result.ticker,
                        company_name=result.company_name,
                        metadata={
                            'document_type': 'skill_shortage_analysis',
                            'skill_shortage_mentions': result.skill_shortage_mentions,
                            'skill_shortage_score': result.skill_shortage_score,
                            'skill_shortage_likelihood': result.skill_shortage_likelihood,
                            'skill_shortage_summary': result.skill_shortage_summary,
                            'year': result.year,
                            'cik': result.cik,
                            'analysis_date': result.analyzed_date
                        }
                    )
                except Exception as e:
                    logger.error(f"Error adding skill shortage analysis to vector store: {e}")

        logger.info("Completed adding skill shortage analyses to vector store")

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