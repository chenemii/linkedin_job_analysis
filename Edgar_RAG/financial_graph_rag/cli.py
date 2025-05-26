"""
Command Line Interface for Financial Vector RAG

Provides easy-to-use commands for setting up and using the system
for M&A organizational structure analysis using vector stores.
"""

import click
import json
import logging
from typing import List, Optional
from pathlib import Path

from .core import FinancialVectorRAG
from .config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Financial Vector RAG - M&A Organizational Structure Analysis Tool"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              default=[2022, 2023],
              help='Years of filings to collect (can specify multiple)')
@click.option('--limit-companies',
              '-c',
              type=int,
              default=None,
              help='Limit number of companies to process')
@click.option('--limit-filings',
              '-f',
              type=int,
              default=2,
              help='Limit filings per company')
@click.option('--force', is_flag=True, help='Force refresh of cached data')
def setup(years, limit_companies, limit_filings, force):
    """Set up the data pipeline: collect S&P 500 data, download filings, build vector stores"""

    click.echo("🚀 Setting up Financial Vector RAG data pipeline...")
    click.echo(f"Years: {list(years)}")
    click.echo(f"Company limit: {limit_companies or 'All'}")
    click.echo(f"Filings per company: {limit_filings}")

    try:
        system = FinancialVectorRAG()

        results = system.setup_data_pipeline(
            years=list(years),
            limit_companies=limit_companies,
            limit_filings_per_company=limit_filings)

        # Display results
        click.echo("\n✅ Data pipeline setup completed!")
        click.echo(f"📊 Companies collected: {results['companies_collected']}")
        click.echo(f"📄 Filings downloaded: {results['filings_downloaded']}")
        click.echo(f"📁 Documents stored: {results['documents_stored']}")
        click.echo(
            f"🗃️ Company collections created: {results['company_collections_created']}"
        )

        if results['errors']:
            click.echo(f"\n⚠️  {len(results['errors'])} errors occurred:")
            for error in results['errors'][:5]:  # Show first 5 errors
                click.echo(f"  - {error}")
            if len(results['errors']) > 5:
                click.echo(
                    f"  ... and {len(results['errors']) - 5} more errors")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during setup: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('company_ticker')
@click.option('--focus',
              '-f',
              default='organizational structure changes',
              help='Analysis focus area')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
@click.option('--format',
              type=click.Choice(['json', 'txt'], case_sensitive=False),
              default='txt',
              help='Output format')
def analyze(company_ticker, focus, output, format):
    """Analyze M&A impact on organizational structure for a specific company"""

    click.echo(f"🔍 Analyzing M&A impact for {company_ticker.upper()}...")
    click.echo(f"Focus: {focus}")

    try:
        system = FinancialVectorRAG()

        results = system.analyze_company_ma_impact(
            company_ticker=company_ticker.upper(), analysis_focus=focus)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            if 'available_companies' in results:
                click.echo("\n📋 Available companies (sample):")
                for ticker in results['available_companies'][:10]:
                    click.echo(f"  - {ticker}")
            raise click.ClickException("Company not found")

        # Display key information
        if 'company_info' in results:
            company = results['company_info']
            click.echo(f"\n🏢 Company: {company['name']} ({company['ticker']})")
            click.echo(f"📍 Sector: {company['sector']}")
            click.echo(f"🏛️  Headquarters: {company['headquarters']}")

        # Display analysis summary
        click.echo("\n📊 Analysis Summary:")
        click.echo(f"Documents analyzed: {results.get('document_count', 0)}")

        analysis = results.get('analysis', 'No analysis available')
        # Show first 500 characters of analysis
        click.echo(analysis[:500] + "..." if len(analysis) > 500 else analysis)

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, format)
            if success:
                click.echo(f"\n💾 Analysis saved to: {output}")
            else:
                click.echo(f"\n❌ Failed to save analysis to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--sector', '-s', help='Focus on specific sector')
@click.option('--companies',
              '-c',
              multiple=True,
              help='Specific company tickers to analyze')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def trends(sector, companies, output):
    """Analyze M&A trends across companies or within a sector"""

    if sector:
        click.echo(f"📈 Analyzing M&A trends in {sector} sector...")
    elif companies:
        click.echo(
            f"📈 Analyzing M&A trends for companies: {', '.join(companies)}")
    else:
        click.echo("📈 Analyzing overall M&A trends...")

    try:
        system = FinancialVectorRAG()

        results = system.get_ma_trends_analysis(
            sector=sector,
            company_tickers=list(companies) if companies else None)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Trends analysis failed")

        # Display results summary
        click.echo(f"\n📊 Trends Analysis Results:")
        click.echo(
            f"Companies searched: {results.get('companies_searched', 0)}")
        click.echo(
            f"Companies with results: {results.get('companies_with_results', 0)}"
        )
        click.echo(f"Total documents: {results.get('total_documents', 0)}")

        # Display sample results
        if 'results_by_company' in results:
            click.echo("\n🏢 Top companies with M&A activity:")
            for ticker, docs in list(
                    results['results_by_company'].items())[:5]:
                click.echo(f"  - {ticker}: {len(docs)} relevant documents")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Trends analysis saved to: {output}")
            else:
                click.echo(f"\n❌ Failed to save trends analysis to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during trends analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('reference_company')
@click.option('--criteria',
              '-c',
              default='organizational impact',
              help='Similarity criteria for matching')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def similar(reference_company, criteria, output):
    """Find companies with similar M&A patterns"""

    click.echo(
        f"🔍 Finding M&A cases similar to {reference_company.upper()}...")
    click.echo(f"Similarity criteria: {criteria}")

    try:
        system = FinancialVectorRAG()

        results = system.search_similar_ma_cases(
            reference_company=reference_company.upper(),
            similarity_criteria=criteria)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Similar cases search failed")

        # Display results
        click.echo(f"\n📊 Similar Cases Analysis:")
        click.echo(
            f"Reference documents: {results.get('reference_documents', 0)}")

        if 'similar_cases' in results:
            similar_data = results['similar_cases']
            click.echo(
                f"Companies analyzed: {similar_data.get('companies_searched', 0)}"
            )
            click.echo(
                f"Companies with similar patterns: {similar_data.get('companies_with_results', 0)}"
            )

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Similar cases analysis saved to: {output}")
            else:
                click.echo(f"\n❌ Failed to save analysis to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during similar cases search: {e}")
        raise click.ClickException(str(e))


@cli.command()
def status():
    """Show system status and statistics"""

    click.echo("🔍 Checking Financial Vector RAG system status...")

    try:
        system = FinancialVectorRAG()
        status_info = system.get_system_status()

        click.echo(f"\n📊 System Status: {status_info['status']}")

        # Vector store status
        if 'vector_store' in status_info:
            vs_status = status_info['vector_store']
            click.echo(f"\n🗃️ Vector Store:")
            click.echo(
                f"  Company collections: {vs_status.get('company_collections', 0)}"
            )
            click.echo(
                f"  Total documents: {vs_status.get('total_documents', 0)}")

            companies_with_data = vs_status.get('companies_with_data', [])
            if companies_with_data:
                click.echo(
                    f"  Companies with data: {', '.join(companies_with_data[:10])}"
                )
                if len(companies_with_data) > 10:
                    click.echo(
                        f"    ... and {len(companies_with_data) - 10} more")

        # S&P 500 data status
        if 'sp500_data' in status_info:
            sp500_status = status_info['sp500_data']
            click.echo(f"\n📈 S&P 500 Data:")
            click.echo(
                f"  Companies loaded: {sp500_status.get('companies_loaded', 0)}"
            )
            click.echo(
                f"  Companies with CIK: {sp500_status.get('companies_with_cik', 0)}"
            )

        # EDGAR data status
        if 'edgar_data' in status_info:
            edgar_status = status_info['edgar_data']
            click.echo(f"\n📄 EDGAR Data:")
            click.echo(
                f"  Filings cached: {edgar_status.get('filings_cached', 0)}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error checking status: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('query')
@click.option('--company', '-c', help='Focus on specific company ticker')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
@click.option('--raw',
              is_flag=True,
              help='Return raw documents without LLM analysis')
def query(query, company, output, raw):
    """Run a custom query against the vector store and get LLM analysis"""

    if company:
        click.echo(f"🔍 Analyzing query for {company.upper()}: {query}")
    else:
        click.echo(f"🔍 Analyzing query across all companies: {query}")

    try:
        system = FinancialVectorRAG()

        if company:
            # Company-specific analysis using LLM
            if raw:
                # Just return documents
                documents = system.rag_engine.retrieve_company_documents(
                    query=query, company_ticker=company.upper(), top_k=5)

                results = {
                    'query': query,
                    'company': company.upper(),
                    'documents': documents,
                    'document_count': len(documents)
                }

                click.echo(f"\n📊 Raw Document Results:")
                click.echo(f"Company: {company.upper()}")
                click.echo(f"Documents found: {len(documents)}")

                for i, doc in enumerate(documents[:3]):
                    click.echo(
                        f"\n📄 Document {i+1} (Score: {doc['score']:.3f}):")
                    click.echo(doc['content'][:200] + "...")
            else:
                # Use LLM analysis
                results = system.rag_engine.analyze_company_ma_impact(
                    query=query, company_ticker=company.upper(), top_k=5)

                if 'error' in results:
                    click.echo(f"❌ {results['error']}")
                    raise click.ClickException("Company analysis failed")

                click.echo(f"\n📊 AI Analysis Results:")
                click.echo(f"Company: {results['company_ticker']}")
                click.echo(
                    f"Chunks analyzed: {results.get('chunk_count', results.get('document_count', 0))}"
                )
                if 'total_chunks_found' in results:
                    click.echo(
                        f"Total chunks found: {results['total_chunks_found']}")
                if 'context_size_chars' in results:
                    click.echo(
                        f"Context size: {results['context_size_chars']:,} characters"
                    )
                click.echo(f"\n🤖 Analysis:")
                click.echo("=" * 60)
                click.echo(results['analysis'])
                click.echo("=" * 60)
        else:
            # Cross-company search with LLM analysis
            if raw:
                # Just return documents
                search_results = system.rag_engine.search_across_companies(
                    query=query, top_k_per_company=3)

                results = search_results

                click.echo(f"\n📊 Raw Cross-Company Results:")
                click.echo(
                    f"Companies searched: {search_results.get('companies_searched', 0)}"
                )
                click.echo(
                    f"Companies with results: {search_results.get('companies_with_results', 0)}"
                )
                click.echo(
                    f"Total documents: {search_results.get('total_documents', 0)}"
                )

                if 'results_by_company' in search_results and search_results[
                        'results_by_company']:
                    for ticker, docs in search_results[
                            'results_by_company'].items():
                        if docs:
                            click.echo(f"\n📈 {ticker}: {len(docs)} documents")
                            for i, doc in enumerate(docs[:2]):
                                click.echo(
                                    f"  📄 Document {i+1} (Score: {doc['score']:.3f}):"
                                )
                                content_preview = doc[
                                    'content'][:300] + "..." if len(
                                        doc['content']
                                    ) > 300 else doc['content']
                                click.echo(f"    {content_preview}")
                else:
                    click.echo(f"\n⚠️ No documents found matching the query.")
            else:
                # Use LLM analysis for cross-company results
                analysis_results = system.analyze_cross_company_query(
                    query=query, top_k_per_company=3)

                results = analysis_results

                if 'error' in analysis_results:
                    click.echo(f"❌ {analysis_results['error']}")
                    raise click.ClickException("Cross-company analysis failed")

                click.echo(f"\n📊 AI Analysis Results:")
                click.echo(
                    f"Companies searched: {analysis_results.get('companies_searched', 0)}"
                )
                click.echo(
                    f"Companies with results: {analysis_results.get('companies_with_results', 0)}"
                )
                click.echo(
                    f"Total chunks found: {analysis_results.get('total_chunks_found', analysis_results.get('total_documents', 0))}"
                )
                if 'chunks_used_in_analysis' in analysis_results:
                    click.echo(
                        f"Chunks used in analysis: {analysis_results['chunks_used_in_analysis']}"
                    )
                if 'context_size_chars' in analysis_results:
                    click.echo(
                        f"Context size: {analysis_results['context_size_chars']:,} characters"
                    )

                click.echo(f"\n🤖 Cross-Company Analysis:")
                click.echo("=" * 60)
                click.echo(analysis_results['analysis'])
                click.echo("=" * 60)

                if 'company_summaries' in analysis_results and analysis_results[
                        'company_summaries']:
                    click.echo(f"\n🏢 Individual Company Insights:")
                    for ticker, summary in analysis_results[
                            'company_summaries'].items():
                        click.echo(f"\n📈 {ticker}:")
                        click.echo(f"  {summary}")

        # Add tip about raw mode
        if not raw:
            click.echo(
                f"\n💡 Tip: Use --raw flag to see original documents instead of AI analysis"
            )

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Analysis results saved to: {output}")
            else:
                click.echo(f"\n❌ Failed to save results to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during query: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    '--company',
    '-c',
    multiple=True,
    help='Specific company ticker(s) to clean (can specify multiple)')
@click.option('--all',
              'clean_all',
              is_flag=True,
              help='Clean all company collections')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def clean(company, clean_all, confirm):
    """Clean vector store collections"""

    if not company and not clean_all:
        click.echo("❌ Please specify either --company TICKER or --all")
        click.echo("Examples:")
        click.echo("  financial-vector-rag clean --company MSFT")
        click.echo(
            "  financial-vector-rag clean --company MSFT --company AAPL")
        click.echo("  financial-vector-rag clean --all")
        raise click.ClickException("No cleaning target specified")

    try:
        system = FinancialVectorRAG()

        if clean_all:
            # Get all available companies
            available_companies = system.rag_engine.get_available_companies()
            companies_to_clean = [
                c['company_ticker'] for c in available_companies
            ]

            if not companies_to_clean:
                click.echo("ℹ️  No company collections found to clean")
                system.close()
                return

            click.echo(
                f"🗑️  Preparing to clean ALL {len(companies_to_clean)} company collections:"
            )
            for ticker in companies_to_clean:
                click.echo(f"  - {ticker}")

        else:
            # Clean specific companies
            companies_to_clean = [c.upper() for c in company]
            click.echo(
                f"🗑️  Preparing to clean {len(companies_to_clean)} company collection(s):"
            )
            for ticker in companies_to_clean:
                click.echo(f"  - {ticker}")

        # Confirmation prompt
        if not confirm:
            if clean_all:
                click.echo(
                    f"\n⚠️  WARNING: This will permanently delete ALL vector store data!"
                )
            else:
                click.echo(
                    f"\n⚠️  WARNING: This will permanently delete vector store data for the specified companies!"
                )

            click.echo("This action cannot be undone.")

            if not click.confirm("Are you sure you want to proceed?"):
                click.echo("❌ Cleaning cancelled")
                system.close()
                return

        # Perform cleaning
        click.echo(f"\n🧹 Starting vector store cleaning...")

        success_count = 0
        error_count = 0
        errors = []

        for ticker in companies_to_clean:
            try:
                success = system.rag_engine.delete_company_collection(ticker)
                if success:
                    click.echo(f"✅ Cleaned collection for {ticker}")
                    success_count += 1
                else:
                    click.echo(
                        f"⚠️  Failed to clean collection for {ticker} (may not exist)"
                    )
                    error_count += 1

            except Exception as e:
                error_msg = f"Error cleaning {ticker}: {e}"
                click.echo(f"❌ {error_msg}")
                errors.append(error_msg)
                error_count += 1

        # Summary
        click.echo(f"\n📊 Cleaning Summary:")
        click.echo(f"  ✅ Successfully cleaned: {success_count} collections")
        if error_count > 0:
            click.echo(f"  ❌ Errors: {error_count} collections")
            if errors:
                click.echo(f"\n🔍 Error details:")
                for error in errors:
                    click.echo(f"  - {error}")

        if success_count > 0:
            click.echo(f"\n🎉 Vector store cleaning completed!")
            if clean_all:
                click.echo(
                    "💡 Tip: Run 'setup' command to rebuild the vector store")
            else:
                click.echo(
                    "💡 Tip: Run 'setup' command to rebuild data for cleaned companies"
                )
        else:
            click.echo(f"\n⚠️  No collections were cleaned")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during cleaning: {e}")
        raise click.ClickException(str(e))


@cli.command()
def examples():
    """Show usage examples"""

    click.echo("📚 Financial Vector RAG Usage Examples")
    click.echo("=" * 50)

    examples = [{
        "title":
        "🚀 Setup Data Pipeline",
        "command":
        "financial-vector-rag setup --years 2023 --limit-companies 10",
        "description":
        "Download and process 10 companies' 2023 filings"
    }, {
        "title": "🔍 Analyze Company M&A Impact",
        "command":
        "financial-vector-rag analyze MSFT --focus 'acquisition integration'",
        "description": "Analyze Microsoft's M&A organizational impact"
    }, {
        "title": "📈 Sector Trends Analysis",
        "command": "financial-vector-rag trends --sector Technology",
        "description": "Analyze M&A trends in technology sector"
    }, {
        "title":
        "🔗 Find Similar M&A Cases",
        "command":
        "financial-vector-rag similar AAPL --criteria 'integration challenges'",
        "description":
        "Find companies with similar M&A integration patterns to Apple"
    }, {
        "title": "📊 Check System Status",
        "command": "financial-vector-rag status",
        "description": "View system status and data statistics"
    }, {
        "title":
        "🔍 Custom Query",
        "command":
        "financial-vector-rag query 'organizational restructuring' --company GOOGL",
        "description":
        "Search Google's documents for organizational restructuring"
    }, {
        "title": "🧹 Clean Specific Company",
        "command": "financial-vector-rag clean --company MSFT",
        "description": "Remove Microsoft's data from vector store"
    }, {
        "title":
        "🗑️ Clean Multiple Companies",
        "command":
        "financial-vector-rag clean --company MSFT --company AAPL",
        "description":
        "Remove multiple companies' data from vector store"
    }, {
        "title":
        "💥 Clean All Data",
        "command":
        "financial-vector-rag clean --all --confirm",
        "description":
        "Remove all vector store data (skip confirmation)"
    }]

    for example in examples:
        click.echo(f"\n{example['title']}")
        click.echo(f"Command: {example['command']}")
        click.echo(f"Description: {example['description']}")

    click.echo(f"\n💡 Tips:")
    click.echo(f"- Use --verbose for detailed logging")
    click.echo(f"- Add --output filename to save results")
    click.echo(f"- Check available companies with: status command")


@cli.command()
@click.argument('company_ticker')
def company_info(company_ticker):
    """Get detailed information about a company's data"""

    click.echo(f"🏢 Getting information for {company_ticker.upper()}...")

    try:
        system = FinancialVectorRAG()

        summary = system.get_company_analysis_summary(company_ticker.upper())

        if 'error' in summary:
            click.echo(f"❌ {summary['error']}")
            raise click.ClickException("Company information retrieval failed")

        # Display company info
        if 'company_info' in summary:
            company = summary['company_info']
            click.echo(f"\n📊 Company Information:")
            click.echo(f"  Name: {company['name']}")
            click.echo(f"  Sector: {company['sector']}")
            click.echo(f"  Headquarters: {company['headquarters']}")
            click.echo(f"  CIK: {company['cik']}")

        # Display vector store stats
        if 'vector_store_stats' in summary:
            stats = summary['vector_store_stats']
            click.echo(f"\n🗃️ Vector Store Statistics:")
            click.echo(f"  Document count: {stats.get('document_count', 0)}")
            click.echo(f"  Status: {stats.get('status', 'unknown')}")

            if 'form_types' in stats:
                click.echo(f"  Form types: {', '.join(stats['form_types'])}")

            if 'avg_ma_score' in stats and stats['avg_ma_score']:
                click.echo(f"  Average M&A score: {stats['avg_ma_score']:.2f}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error getting company information: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    cli()
