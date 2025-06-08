"""
Core Financial Vector RAG System

Main orchestrator class that coordinates data collection and vector store building
for M&A analysis using company-specific vector collections.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json

from .config import settings
from .data_collectors import SP500Collector, EdgarFilingCollector, SP500Company, EdgarFiling, SkillShortageAnalyzer
from .rag_engine.vector_rag import VectorRAGEngine
from .rag_engine.skill_shortage_rag import SkillShortageRAGEngine

logger = logging.getLogger(__name__)


class FinancialVectorRAG:
    """
    Main Financial Vector RAG system for M&A organizational structure analysis and skill shortage analysis
    """

    def __init__(self):
        """Initialize the Financial Vector RAG system"""
        logger.info("Initializing Financial Vector RAG system")

        # Initialize components
        self.sp500_collector = SP500Collector()
        self.edgar_collector = EdgarFilingCollector()
        self.rag_engine = VectorRAGEngine()
        self.skill_shortage_analyzer = SkillShortageAnalyzer()
        self.skill_shortage_rag = SkillShortageRAGEngine()

        logger.info("Financial Vector RAG system initialized")

    def setup_data_pipeline(
            self,
            years: List[int] = [2022, 2023],
            limit_companies: Optional[int] = None,
            limit_filings_per_company: Optional[int] = 2) -> Dict:
        """
        Set up the complete data pipeline from S&P 500 collection to vector stores
        
        Args:
            years: Years of filings to collect
            limit_companies: Optional limit on number of companies
            limit_filings_per_company: Optional limit on filings per company
            
        Returns:
            Pipeline execution results
        """
        logger.info("Starting data pipeline setup")

        results = {
            'companies_collected': 0,
            'filings_downloaded': 0,
            'documents_stored': 0,
            'company_collections_created': 0,
            'errors': []
        }

        try:
            # Step 1: Collect S&P 500 companies
            logger.info("Step 1: Collecting S&P 500 companies")
            companies = self.sp500_collector.collect_all()
            companies_with_cik = self.sp500_collector.get_companies_with_cik()

            if limit_companies:
                companies_with_cik = companies_with_cik[:limit_companies]

            results['companies_collected'] = len(companies_with_cik)
            logger.info(
                f"Collected {len(companies_with_cik)} companies with CIK numbers"
            )

            # Step 2: Download 10-K filings
            logger.info("Step 2: Downloading EDGAR 10-K filings")
            filings = self.edgar_collector.download_10k_filings(
                companies_with_cik, years, limit_filings_per_company)

            results['filings_downloaded'] = len(filings)
            logger.info(f"Downloaded {len(filings)} filings")

            # Filter for M&A relevant filings
            ma_relevant_filings = self.edgar_collector.filter_ma_relevant_filings(
            )
            logger.info(
                f"Found {len(ma_relevant_filings)} M&A relevant filings")

            # Step 3: Store documents in company-specific vector stores
            logger.info(
                "Step 3: Storing documents in company-specific vector stores")
            company_collections = set()

            for filing in ma_relevant_filings:
                try:
                    if filing.text_content:
                        # Get company info
                        company = next((c for c in companies_with_cik
                                        if c.cik == filing.cik), None)
                        company_ticker = company.symbol if company else filing.cik
                        company_name = company.name if company else filing.company_name

                        # Add document to company-specific vector store
                        self.rag_engine.add_document_to_company_store(
                            document_text=filing.text_content,
                            document_id=filing.accession_number,
                            company_ticker=company_ticker,
                            company_name=company_name,
                            metadata={
                                'company_name': filing.company_name,
                                'cik': filing.cik,
                                'filing_date': filing.filing_date,
                                'form_type': filing.form_type,
                                'ma_score': filing.ma_score,
                                'has_ma_content': filing.has_ma_content
                            })

                        company_collections.add(company_ticker)
                        results['documents_stored'] += 1

                except Exception as e:
                    error_msg = f"Error processing filing {filing.accession_number}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            results['company_collections_created'] = len(company_collections)

            # Step 4: Cache results
            self.sp500_collector._cache_companies(companies)
            self.edgar_collector.save_cache()

            logger.info("Data pipeline setup completed successfully")

        except Exception as e:
            error_msg = f"Error in data pipeline setup: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        return results

    def analyze_company_ma_impact(
            self,
            company_ticker: str,
            analysis_focus: str = "organizational structure changes") -> Dict:
        """
        Analyze M&A impact on organizational structure for a specific company
        
        Args:
            company_ticker: Company ticker symbol
            analysis_focus: Specific aspect to analyze
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Analyzing M&A impact for {company_ticker}")

        # Get company information
        company = self.sp500_collector.get_company_by_symbol(company_ticker)
        if not company:
            available_companies = self.rag_engine.get_available_companies()
            return {
                'error':
                f"Company {company_ticker} not found in S&P 500",
                'available_companies':
                [c['company_ticker'] for c in available_companies]
            }

        # Formulate analysis query
        query = f"Analyze {analysis_focus} for {company.name} ({company_ticker}) due to mergers and acquisitions"

        # Use Vector RAG engine for analysis
        results = self.rag_engine.analyze_company_ma_impact(
            query=query, company_ticker=company_ticker)

        # Add company context
        results['company_info'] = {
            'name': company.name,
            'ticker': company.symbol,
            'cik': company.cik,
            'sector': company.sector,
            'headquarters': company.headquarters
        }

        return results

    def get_ma_trends_analysis(self,
                               sector: Optional[str] = None,
                               company_tickers: List[str] = None) -> Dict:
        """
        Analyze M&A trends across companies or within a sector
        
        Args:
            sector: Optional sector filter
            company_tickers: Optional list of specific companies to analyze
            
        Returns:
            M&A trends analysis results
        """
        logger.info("Performing M&A trends analysis")

        # Get companies to analyze
        if company_tickers:
            target_companies = company_tickers
        elif sector:
            # Get companies in specific sector
            companies = self.sp500_collector.collect_all()
            target_companies = [
                c.symbol for c in companies if c.sector == sector
            ]
        else:
            # Get all available companies with data
            available = self.rag_engine.get_available_companies()
            target_companies = [c['company_ticker'] for c in available]

        # Search for M&A trends across companies
        query = "mergers acquisitions organizational structure changes trends"

        results = self.rag_engine.search_across_companies(
            query=query, company_tickers=target_companies, top_k_per_company=3)

        # Add sector context if specified
        if sector:
            results['sector'] = sector

        results['analysis_type'] = 'trends'
        return results

    def search_similar_ma_cases(
            self,
            reference_company: str,
            similarity_criteria: str = "organizational impact") -> Dict:
        """
        Search for companies with similar M&A patterns
        
        Args:
            reference_company: Reference company ticker
            similarity_criteria: Criteria for similarity matching
            
        Returns:
            Similar M&A cases analysis
        """
        logger.info(f"Searching for M&A cases similar to {reference_company}")

        # Get reference company documents
        reference_docs = self.rag_engine.retrieve_company_documents(
            query=f"mergers acquisitions {similarity_criteria}",
            company_ticker=reference_company,
            top_k=5)

        if not reference_docs:
            return {
                'error':
                f"No M&A data found for reference company {reference_company}",
                'reference_company': reference_company
            }

        # Extract key terms from reference documents for similarity search
        reference_text = " ".join(
            [doc['content'][:500] for doc in reference_docs])

        # Search across all companies
        results = self.rag_engine.search_across_companies(
            query=reference_text[:1000],  # Limit query size
            top_k_per_company=2)

        return {
            'reference_company': reference_company,
            'similarity_criteria': similarity_criteria,
            'reference_documents': len(reference_docs),
            'similar_cases': results
        }

    def analyze_cross_company_query(self,
                                    query: str,
                                    top_k_per_company: int = 3) -> Dict:
        """
        Analyze a query across multiple companies using top-ranked chunks from all companies
        
        Args:
            query: Query to analyze
            top_k_per_company: Number of chunks to retrieve per company
            
        Returns:
            Cross-company analysis results with LLM insights based on top-ranked chunks
        """
        logger.info(f"Analyzing cross-company query: {query}")

        try:
            # Get chunks from all companies
            search_results = self.rag_engine.search_across_companies(
                query=query, top_k_per_company=top_k_per_company)

            if search_results.get('total_documents', 0) == 0:
                return {
                    'error':
                    'No documents found matching the query across any companies.',
                    'query':
                    query,
                    'companies_searched':
                    search_results.get('companies_searched', 0),
                    'suggestion':
                    'Try simpler terms like "merger", "acquisition", or "restructuring"'
                }

            # Get the globally top-ranked chunks across all companies
            top_global_chunks = search_results.get('top_results_global', [])

            # Prepare context for LLM analysis using the highest-ranked chunks
            max_context_chars = 10000  # Conservative limit for cross-company analysis
            current_chars = 0
            selected_chunks = []
            company_chunk_counts = {}

            # Select chunks based on global ranking, ensuring diversity across companies
            for chunk in top_global_chunks:
                company = chunk['company_ticker']
                chunk_size = len(chunk['content'])

                # Track chunks per company for diversity
                if company not in company_chunk_counts:
                    company_chunk_counts[company] = 0

                # Include chunk if it fits in context and maintains reasonable diversity
                if (current_chars + chunk_size <= max_context_chars
                        and company_chunk_counts[company]
                        < 3):  # Max 3 chunks per company in context

                    selected_chunks.append(chunk)
                    current_chars += chunk_size
                    company_chunk_counts[company] += 1

                    # Stop if we have enough context
                    if len(selected_chunks) >= 15:  # Max 15 chunks total
                        break

            # Generate individual company summaries using their top chunks
            company_summaries = {}
            for ticker, chunks in search_results.get('results_by_company',
                                                     {}).items():
                if chunks:
                    # Use top 2 chunks from each company for individual summary
                    company_context = "\n\n".join([
                        f"Chunk (Score: {chunk['score']:.3f}): {chunk['content'][:800]}..."
                        for chunk in chunks[:2]
                    ])

                    individual_prompt = f"""Based on the following top-ranked document chunks from {ticker}, provide a concise summary related to the query "{query}":

{company_context}

Provide a 2-3 sentence summary of key findings for {ticker}:"""

                    try:
                        individual_response = self.rag_engine.llm.invoke(
                            individual_prompt)
                        company_summaries[
                            ticker] = individual_response.content.strip()
                    except Exception as e:
                        logger.warning(
                            f"Error generating summary for {ticker}: {e}")
                        company_summaries[
                            ticker] = f"Unable to generate summary for {ticker}"

            # Create comprehensive analysis prompt using globally top-ranked chunks
            combined_context = "\n\n".join([
                f"Chunk {i+1} from {chunk['company_ticker']} (Similarity Score: {chunk['score']:.3f}):\n{chunk['content']}"
                for i, chunk in enumerate(selected_chunks)
            ])

            analysis_prompt = f"""You are a financial analyst specializing in M&A activities. Analyze the following document chunks from multiple companies to answer this query: "{query}"

The chunks below are ranked by similarity to your query across all companies. Use these high-quality, relevant chunks to provide comprehensive insights.

Top-ranked document chunks:
{combined_context}

Please provide a comprehensive analysis that:
1. Directly answers the query "{query}" using evidence from the chunks
2. Identifies common patterns and trends across companies
3. Highlights notable differences between companies  
4. Provides actionable insights based on the findings
5. Uses specific examples from the document chunks when possible
6. Acknowledges the similarity scores when discussing relevance

Format your response as a well-structured analysis with clear sections and bullet points where appropriate.
Focus on the most relevant information from the highest-scoring chunks."""

            # Generate LLM analysis
            try:
                response = self.rag_engine.llm.invoke(analysis_prompt)
                analysis = response.content
            except Exception as e:
                logger.error(f"Error generating LLM analysis: {e}")
                analysis = f"Error generating analysis: {str(e)}"

            # Compile results with detailed chunk information
            results = {
                'query':
                query,
                'analysis':
                analysis,
                'companies_searched':
                search_results.get('companies_searched', 0),
                'companies_with_results':
                search_results.get('companies_with_results', 0),
                'total_chunks_found':
                search_results.get('total_documents', 0),
                'chunks_used_in_analysis':
                len(selected_chunks),
                'context_size_chars':
                current_chars,
                'company_summaries':
                company_summaries,
                'chunk_distribution':
                company_chunk_counts,
                'top_chunks_used': [{
                    'company': chunk['company_ticker'],
                    'chunk_id': chunk['id'],
                    'similarity_score': chunk['score'],
                    'chunk_size': len(chunk['content'])
                } for chunk in selected_chunks],
                'analysis_type':
                'cross_company_query'
            }

            return results

        except Exception as e:
            logger.error(f"Error in cross-company query analysis: {e}")
            return {
                'error': str(e),
                'query': query,
                'analysis_type': 'cross_company_query'
            }

    def export_analysis_report(self,
                               analysis_results: Dict,
                               output_path: str,
                               format: str = "json") -> bool:
        """
        Export analysis results to file
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Output file path
            format: Export format ('json' or 'txt')
            
        Returns:
            True if export successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results,
                              f,
                              indent=2,
                              ensure_ascii=False)

            elif format.lower() == "txt":
                with open(output_file, 'w', encoding='utf-8') as f:
                    # Write formatted text report
                    f.write("Financial M&A Analysis Report\n")
                    f.write("=" * 50 + "\n\n")

                    if 'company_info' in analysis_results:
                        company = analysis_results['company_info']
                        f.write(
                            f"Company: {company['name']} ({company['ticker']})\n"
                        )
                        f.write(f"Sector: {company.get('sector', 'N/A')}\n")
                        f.write(
                            f"Headquarters: {company.get('headquarters', 'N/A')}\n\n"
                        )

                    if 'analysis' in analysis_results:
                        f.write("Analysis:\n")
                        f.write("-" * 20 + "\n")
                        f.write(analysis_results['analysis'])
                        f.write("\n\n")

                    if 'document_count' in analysis_results:
                        f.write(
                            f"Documents analyzed: {analysis_results['document_count']}\n"
                        )

                    # Add metadata
                    f.write(
                        f"\nGenerated: {json.dumps(analysis_results.get('query', ''), indent=2)}\n"
                    )

            logger.info(f"Analysis report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            return False

    def get_system_status(self) -> Dict:
        """
        Get system status and statistics
        
        Returns:
            System status information
        """
        try:
            # Get vector store status
            vector_status = self._get_vector_store_status()

            # Get S&P 500 collector status
            sp500_status = {
                'companies_loaded':
                len(self.sp500_collector.companies),
                'companies_with_cik':
                len(self.sp500_collector.get_companies_with_cik())
            }

            # Get EDGAR collector status
            edgar_status = {
                'filings_cached': len(self.edgar_collector.filings),
                'cache_directory': str(self.edgar_collector.filings_cache)
            }

            return {
                'status': 'operational',
                'vector_store': vector_status,
                'sp500_data': sp500_status,
                'edgar_data': edgar_status,
                'system_type': 'vector_rag'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'system_type': 'vector_rag'
            }

    def _get_vector_store_status(self) -> Dict:
        """Get vector store status"""
        try:
            companies = self.rag_engine.get_available_companies()

            total_documents = sum(c['document_count'] for c in companies)

            return {
                'company_collections':
                len(companies),
                'total_documents':
                total_documents,
                'companies_with_data': [
                    c['company_ticker'] for c in companies
                    if c['document_count'] > 0
                ],
                'status':
                'connected'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def close(self):
        """Close all connections and cleanup resources"""
        try:
            self.rag_engine.close()
            logger.info("Financial Vector RAG system closed")
        except Exception as e:
            logger.error(f"Error closing system: {e}")

    def get_company_analysis_summary(self, company_ticker: str) -> Dict:
        """
        Get summary of available analysis data for a company
        
        Args:
            company_ticker: Company ticker symbol
            
        Returns:
            Company analysis summary
        """
        try:
            # Get company statistics
            stats = self.rag_engine.get_company_statistics(company_ticker)

            # Get company info from S&P 500 data
            company = self.sp500_collector.get_company_by_symbol(
                company_ticker)

            result = {
                'company_ticker': company_ticker,
                'vector_store_stats': stats
            }

            if company:
                result['company_info'] = {
                    'name': company.name,
                    'sector': company.sector,
                    'headquarters': company.headquarters,
                    'cik': company.cik
                }

            return result

        except Exception as e:
            logger.error(
                f"Error getting company analysis summary for {company_ticker}: {e}"
            )
            return {'company_ticker': company_ticker, 'error': str(e)}

    def get_company_info(self, ticker: str, detailed: bool = False) -> Dict:
        """
        Get information about a specific company
        
        Args:
            ticker: Company ticker symbol
            detailed: Whether to include detailed information
            
        Returns:
            Company information dictionary
        """
        try:
            # Get company from S&P 500 data
            company = self.sp500_collector.get_company_by_symbol(ticker)
            
            if not company:
                return {
                    'error': f'Company {ticker} not found in S&P 500 data'
                }
            
            # Get vector store statistics
            stats = self.rag_engine.get_company_statistics(ticker)
            
            result = {
                'ticker': company.symbol,
                'name': company.name,
                'sector': company.sector,
                'headquarters': company.headquarters,
                'document_count': stats.get('document_count', 0)
            }
            
            if detailed:
                result['detailed_info'] = {
                    'cik': company.cik,
                    'sub_industry': company.sub_industry,
                    'date_added': company.date_added,
                    'founded': company.founded,
                    'vector_store_stats': stats,
                    'recent_ma_events': []  # Placeholder for M&A events
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {e}")
            return {'error': str(e)}

    def cleanup_system(self) -> Dict:
        """
        Clean up temporary files and reset the system
        
        Returns:
            Cleanup results
        """
        try:
            import shutil
            from pathlib import Path
            
            files_removed = 0
            space_freed = 0
            
            # Clean up cache directories
            cache_dirs = [
                Path(settings.cache_directory),
                Path(settings.chroma_persist_directory),
            ]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    # Calculate size before deletion
                    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    space_freed += total_size
                    
                    # Count files
                    file_count = len(list(cache_dir.rglob('*')))
                    files_removed += file_count
                    
                    # Remove directory
                    shutil.rmtree(cache_dir)
                    
            # Recreate cache directory
            Path(settings.cache_directory).mkdir(parents=True, exist_ok=True)
            Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Convert bytes to MB
            space_freed_mb = space_freed / (1024 * 1024)
            
            return {
                'files_removed': files_removed,
                'space_freed': f'{space_freed_mb:.2f} MB'
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}

    def analyze_company_skill_shortage(self, 
                                     company_ticker: str,
                                     analysis_focus: str = "comprehensive skill shortage analysis") -> Dict:
        """
        Analyze skill shortage issues for a specific company
        
        Args:
            company_ticker: Company ticker symbol
            analysis_focus: Specific aspect to analyze
            
        Returns:
            Comprehensive skill shortage analysis results
        """
        logger.info(f"Analyzing skill shortage for {company_ticker}")

        # Get company information
        company = self.sp500_collector.get_company_by_symbol(company_ticker)
        if not company:
            available_companies = self.rag_engine.get_available_companies()
            return {
                'error': f"Company {company_ticker} not found in S&P 500",
                'available_companies': [c['company_ticker'] for c in available_companies]
            }

        # Use Skill Shortage RAG engine for analysis
        results = self.skill_shortage_rag.analyze_company_skill_shortage(
            company_ticker=company_ticker,
            analysis_focus=analysis_focus
        )

        # Add company context
        results['company_info'] = {
            'name': company.name,
            'ticker': company.symbol,
            'cik': company.cik,
            'sector': company.sector,
            'headquarters': company.headquarters
        }

        return results

    def run_skill_shortage_analysis_pipeline(self,
                                           years: List[int] = [2022, 2023],
                                           limit_companies: Optional[int] = None,
                                           save_results: bool = True) -> Dict:
        """
        Run the complete skill shortage analysis pipeline
        
        Args:
            years: Years of filings to analyze
            limit_companies: Optional limit on number of companies
            save_results: Whether to save results to files
            
        Returns:
            Pipeline execution results
        """
        logger.info("Starting skill shortage analysis pipeline")

        results = {
            'companies_analyzed': 0,
            'filings_analyzed': 0,
            'skill_shortage_findings': 0,
            'errors': []
        }

        try:
            # Step 1: Get companies and filings (reuse existing data if available)
            companies = self.sp500_collector.collect_all()
            companies_with_cik = self.sp500_collector.get_companies_with_cik()

            if limit_companies:
                companies_with_cik = companies_with_cik[:limit_companies]

            # Load existing filings if available
            if not self.edgar_collector.load_cache():
                logger.info("No cached filings found, downloading new filings")
                filings = self.edgar_collector.download_10k_filings(
                    companies_with_cik, years, limit_filings_per_company=2)
            else:
                logger.info("Using cached filings")
                filings = self.edgar_collector.filings

            results['companies_analyzed'] = len(companies_with_cik)
            results['filings_analyzed'] = len(filings)

            # Step 2: Run skill shortage analysis
            logger.info("Running skill shortage analysis on filings")
            skill_shortage_results = self.skill_shortage_analyzer.analyze_edgar_filings(
                filings=filings,
                companies=companies_with_cik
            )

            # Count significant findings
            significant_findings = self.skill_shortage_analyzer.filter_significant_findings()
            results['skill_shortage_findings'] = len(significant_findings)

            # Step 3: Add to vector store for enhanced retrieval
            logger.info("Adding skill shortage analysis to vector store")
            self.skill_shortage_rag.add_skill_shortage_analysis_to_vector_store(
                skill_shortage_results
            )

            # Step 4: Save results if requested
            if save_results:
                output_path = self.skill_shortage_analyzer.save_results()
                self.skill_shortage_analyzer.save_cache()
                results['output_file'] = output_path

            logger.info("Skill shortage analysis pipeline completed successfully")

        except Exception as e:
            error_msg = f"Error in skill shortage analysis pipeline: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        return results

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

        return self.skill_shortage_rag.compare_skill_shortage_across_companies(
            company_tickers=company_tickers,
            sector=sector,
            analysis_focus=analysis_focus
        )

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

        return self.skill_shortage_rag.analyze_skill_shortage_trends(
            years=years,
            sector=sector
        )

    def get_skill_shortage_summary_stats(self) -> Dict:
        """
        Get summary statistics about skill shortage analysis
        
        Returns:
            Summary statistics
        """
        return self.skill_shortage_analyzer.get_summary_stats()

    def get_system_statistics(self) -> Dict:
        """
        Get comprehensive system statistics
        
        Returns:
            System statistics dictionary
        """
        try:
            # Get basic system status
            status = self.get_system_status()
            
            # Get available companies
            companies = self.rag_engine.get_available_companies()
            
            # Calculate additional statistics
            total_documents = sum(c.get('document_count', 0) for c in companies)
            
            # Get years covered from document metadata
            years_covered = set()
            for company_data in companies:
                company_ticker = company_data['company_ticker']
                try:
                    stats = self.rag_engine.get_company_statistics(company_ticker)
                    sample_dates = stats.get('sample_filing_dates', [])
                    for date in sample_dates:
                        if isinstance(date, str) and len(date) >= 4:
                            year = date[:4]
                            if year.isdigit():
                                years_covered.add(int(year))
                except:
                    continue
            
            return {
                'total_companies': len(companies),
                'total_documents': total_documents,
                'total_embeddings': total_documents,  # Approximate
                'years_covered': sorted(list(years_covered)),
                'companies_with_data': [c['company_ticker'] for c in companies if c.get('document_count', 0) > 0],
                'system_status': status.get('status', 'unknown'),
                'recent_activity': [
                    f"Vector store contains {len(companies)} company collections",
                    f"Total of {total_documents} documents indexed",
                    f"Covering years: {min(years_covered) if years_covered else 'N/A'} - {max(years_covered) if years_covered else 'N/A'}",
                    f"Data sources: S&P 500, EDGAR filings"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            return {'error': str(e)}

    def analyze_trends(self, 
                      years: List[int] = None,
                      sector: str = None,
                      focus_area: str = "M&A trends") -> Dict:
        """
        Analyze trends in M&A or skill shortage data
        
        Args:
            years: Optional list of years to analyze
            sector: Optional sector filter
            focus_area: Type of trends to analyze
            
        Returns:
            Trend analysis results
        """
        logger.info(f"Analyzing trends: {focus_area}")
        
        try:
            if "skill" in focus_area.lower():
                # Use skill shortage trend analysis
                return self.analyze_skill_shortage_trends(years=years, sector=sector)
            else:
                # Use M&A trend analysis
                return self.get_ma_trends_analysis(sector=sector)
                
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'error': str(e), 'focus_area': focus_area}

    def get_skill_shortage_statistics(self) -> Dict:
        """
        Get skill shortage analysis statistics
        
        Returns:
            Skill shortage statistics
        """
        try:
            # Get basic stats from analyzer
            basic_stats = self.get_skill_shortage_summary_stats()
            
            # Get companies with skill shortage data
            companies = self.rag_engine.get_available_companies()
            
            # Calculate additional metrics
            total_companies_analyzed = len([c for c in companies if c.get('document_count', 0) > 0])
            
            return {
                'total_companies_analyzed': total_companies_analyzed,
                'total_skill_shortage_mentions': basic_stats.get('total_mentions', 0),
                'avg_severity_score': basic_stats.get('average_severity', 0),
                'most_affected_sectors': basic_stats.get('top_sectors', []),
                'recent_findings': [
                    f"Analyzed {total_companies_analyzed} companies",
                    f"Found {basic_stats.get('total_mentions', 0)} skill shortage mentions",
                    f"Average severity: {basic_stats.get('average_severity', 0):.2f}/10",
                    "Skill shortage data available in vector store",
                    "Ready for detailed company analysis"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting skill shortage statistics: {e}")
            return {'error': str(e)}

    def analyze_from_csv_data(self, 
                            csv_path: str,
                            limit: Optional[int] = None) -> Dict:
        """
        Analyze skill shortage from CSV data (similar to the original provided code)
        
        Args:
            csv_path: Path to CSV file with columns: cik, Year, FName, gvkey
            limit: Optional limit on number of filings to process
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing skill shortage from CSV data: {csv_path}")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Run analysis
            results = self.skill_shortage_analyzer.analyze_from_urls(df, limit=limit)
            
            # Save results
            output_path = self.skill_shortage_analyzer.save_results()
            self.skill_shortage_analyzer.save_cache()
            
            # Get summary stats
            stats = self.skill_shortage_analyzer.get_summary_stats()
            
            return {
                'analysis_results': len(results),
                'significant_findings': len([r for r in results if r.skill_shortage_mentions > 0]),
                'output_file': output_path,
                'summary_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSV data: {e}")
            return {'error': str(e)}
