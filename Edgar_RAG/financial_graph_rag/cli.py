"""
Financial Graph RAG CLI

Command-line interface for the Financial Graph RAG system.
Provides commands for data collection, analysis, and querying.
"""

import click
import logging
from pathlib import Path
from typing import Optional, List

from .core import FinancialVectorRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """Financial Graph RAG - Analyze M&A organizational impacts and skill shortages in financial filings"""
    pass


@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to collect data for (can specify multiple)')
@click.option('--limit',
              '-l',
              type=int,
              help='Limit number of companies to process')
@click.option('--force',
              '-f',
              is_flag=True,
              help='Force re-download of existing data')
def setup(years, limit, force):
    """Setup the data pipeline and collect initial data"""

    if years:
        click.echo(f"🚀 Setting up data pipeline for years: {list(years)}")
    else:
        click.echo("🚀 Setting up data pipeline for all available years...")

    if limit:
        click.echo(f"Processing limit: {limit} companies")

    try:
        system = FinancialVectorRAG()

        # Setup data pipeline
        results = system.setup_data_pipeline(
            years=list(years) if years else [2022, 2023],
            limit_companies=limit)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Setup failed")

        click.echo(f"\n✅ Setup completed successfully!")
        click.echo(f"📊 Companies collected: {results.get('companies_collected', 0)}")
        click.echo(f"📄 Filings downloaded: {results.get('filings_downloaded', 0)}")
        click.echo(f"🔍 Documents stored: {results.get('documents_stored', 0)}")
        click.echo(f"🏢 Company collections created: {results.get('company_collections_created', 0)}")

        if results.get('errors'):
            click.echo(f"\n⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                click.echo(f"  - {error}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during setup: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to analyze (can specify multiple)')
@click.option('--limit',
              '-l',
              type=int,
              help='Limit number of companies to analyze')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def analyze(years, limit, output):
    """Run M&A organizational impact analysis"""

    if years:
        click.echo(f"📈 Analyzing M&A impacts for years: {list(years)}")
    else:
        click.echo("📈 Analyzing M&A impacts for all available years...")

    if limit:
        click.echo(f"Analysis limit: {limit} companies")

    try:
        system = FinancialVectorRAG()

        # Get M&A trends analysis instead
        results = system.get_ma_trends_analysis()

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Analysis failed")

        # Display results summary
        click.echo(f"\n📊 Analysis Results:")
        click.echo(f"Companies analyzed: {results.get('companies_analyzed', 0)}")
        click.echo(f"M&A events found: {results.get('ma_events_found', 0)}")
        click.echo(f"Organizational impacts identified: {results.get('organizational_impacts', 0)}")

        # Display key findings
        if 'key_findings' in results:
            click.echo(f"\n🎯 Key Findings:")
            for finding in results['key_findings'][:5]:  # Show top 5
                click.echo(f"  - {finding}")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Analysis saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to analyze (can specify multiple)')
@click.option('--sector', '-s', help='Focus on specific sector')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def trends(years, sector, output):
    """Analyze trends in M&A organizational impacts"""

    if years:
        click.echo(f"📈 Analyzing trends for years: {list(years)}")
    else:
        click.echo("📈 Analyzing trends across all available years...")

    if sector:
        click.echo(f"Focusing on {sector} sector")

    try:
        system = FinancialVectorRAG()

        results = system.get_ma_trends_analysis(sector=sector)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Trends analysis failed")

        # Display results summary
        click.echo(f"\n📊 Trends Analysis Results:")
        click.echo(f"Data points: {results.get('data_points', 0)}")
        click.echo(f"Years covered: {results.get('years_covered', [])}")
        click.echo(f"Companies covered: {results.get('companies_covered', 0)}")

        # Display trend data
        if 'trend_analysis' in results:
            trend_data = results['trend_analysis']
            if 'yearly_trends' in trend_data:
                click.echo("\n📅 Yearly Trends:")
                for year, data in sorted(trend_data['yearly_trends'].items()):
                    click.echo(f"  {year}: {data['total_events']} events, "
                             f"{data['avg_impact_score']:.2f} avg impact")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Trends analysis saved to: {output}")

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
    """Find companies with similar M&A organizational impacts"""

    click.echo(
        f"🔍 Finding companies similar to {reference_company.upper()}...")
    click.echo(f"Criteria: {criteria}")

    try:
        system = FinancialVectorRAG()

        results = system.search_similar_ma_cases(
            reference_company=reference_company.upper(),
            similarity_criteria=criteria)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Similar companies search failed")

        # Display results
        click.echo(f"\n🎯 Similar Companies Found:")
        if 'similar_companies' in results:
            for i, company in enumerate(results['similar_companies'][:10], 1):
                click.echo(f"  {i}. {company.get('company_name', 'Unknown')} "
                         f"({company.get('ticker', 'N/A')}) - "
                         f"Similarity: {company.get('similarity_score', 0):.2f}")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Results saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error finding similar companies: {e}")
        raise click.ClickException(str(e))


@cli.command()
def status():
    """Show system status and statistics"""

    click.echo("📊 Financial Graph RAG System Status")

    try:
        system = FinancialVectorRAG()

        stats = system.get_system_statistics()

        if 'error' in stats:
            click.echo(f"❌ {stats['error']}")
            raise click.ClickException("Failed to get system status")

        # Display statistics
        click.echo(f"\n📈 Data Statistics:")
        click.echo(f"Total companies: {stats.get('total_companies', 0)}")
        click.echo(f"Total documents: {stats.get('total_documents', 0)}")
        click.echo(f"Total embeddings: {stats.get('total_embeddings', 0)}")
        click.echo(f"Years covered: {stats.get('years_covered', [])}")

        # Display recent activity
        if 'recent_activity' in stats:
            click.echo(f"\n🕒 Recent Activity:")
            for activity in stats['recent_activity'][:5]:
                click.echo(f"  - {activity}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error getting statistics: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('query')
@click.option('--limit',
              '-l',
              type=int,
              default=5,
              help='Number of results to return')
@click.option('--context',
              '-c',
              is_flag=True,
              help='Include additional context in results')
def query(query, limit, context):
    """Query the system with natural language"""

    click.echo(f"🔍 Searching for: {query}")

    try:
        system = FinancialVectorRAG()

        # Use the correct method name
        results = system.analyze_cross_company_query(
            query=query,
            top_k_per_company=limit)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Query failed")

        # Display results
        click.echo(f"\n📋 Query Results:")
        click.echo(f"Companies searched: {results.get('companies_searched', 0)}")
        click.echo(f"Companies with results: {results.get('companies_with_results', 0)}")
        click.echo(f"Total chunks found: {results.get('total_chunks_found', 0)}")
        
        # Display analysis
        if 'analysis' in results:
            click.echo(f"\n📄 Analysis:")
            click.echo(results['analysis'])
        
        # Display company summaries if available
        if context and 'company_summaries' in results:
            click.echo(f"\n📊 Company Summaries:")
            for company, summary in results['company_summaries'].items():
                click.echo(f"\n{company}: {summary}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during query: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--confirm',
              '-c',
              is_flag=True,
              help='Confirm deletion without prompting')
def clean(confirm):
    """Clean up temporary files and reset the system"""

    if not confirm:
        if not click.confirm("⚠️  This will delete all cached data. Continue?"):
            click.echo("Operation cancelled.")
            return

    click.echo("🧹 Cleaning up system data...")

    try:
        system = FinancialVectorRAG()

        results = system.cleanup_system()

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Cleanup failed")

        click.echo(f"✅ Cleanup completed!")
        click.echo(f"Files removed: {results.get('files_removed', 0)}")
        click.echo(f"Space freed: {results.get('space_freed', '0 MB')}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e}")
        raise click.ClickException(str(e))


@cli.command()
def examples():
    """Show usage examples"""

    click.echo("📚 Financial Graph RAG Usage Examples\n")

    examples = [
        ("Setup data pipeline", "financial-rag setup --years 2020 2021 --limit 50"),
        ("Analyze M&A impacts", "financial-rag analyze --years 2021 --output results.json"),
        ("View trends", "financial-rag trends --sector technology"),
        ("Find similar companies", "financial-rag similar AAPL --criteria 'organizational impact'"),
        ("Query system", "financial-rag query 'workforce reduction after merger'"),
        ("Check status", "financial-rag status"),
        ("Skill shortage analysis", "financial-rag skill-shortage-pipeline --years 2020 2021"),
        ("Company skill analysis", "financial-rag skill-shortage-company AAPL"),
        ("Skill shortage trends", "financial-rag skill-shortage-trends --sector tech"),
        ("Hiring difficulties pipeline", "financial-rag hiring-difficulties-pipeline --years 2022 2023"),
        ("Company hiring analysis", "financial-rag hiring-difficulties-company AAPL"),
        ("Hiring difficulties rankings", "financial-rag hiring-difficulties-rankings --recent-only"),
        ("Hiring difficulties stats", "financial-rag hiring-difficulties-stats"),
    ]

    for description, command in examples:
        click.echo(f"• {description}:")
        click.echo(f"  {command}\n")


@cli.command()
@click.argument('ticker')
@click.option('--detailed',
              '-d',
              is_flag=True,
              help='Show detailed company information')
def company_info(ticker, detailed):
    """Get information about a specific company"""

    click.echo(f"🏢 Getting information for {ticker.upper()}...")

    try:
        system = FinancialVectorRAG()

        info = system.get_company_info(ticker=ticker.upper(),
                                     detailed=detailed)

        if 'error' in info:
            click.echo(f"❌ {info['error']}")
            raise click.ClickException("Failed to get company information")

        # Display basic info
        click.echo(f"\n📊 Company: {info.get('name', 'Unknown')}")
        click.echo(f"Ticker: {info.get('ticker', ticker.upper())}")
        click.echo(f"Sector: {info.get('sector', 'Unknown')}")
        click.echo(f"Documents available: {info.get('document_count', 0)}")

        # Display detailed info if requested
        if detailed and 'detailed_info' in info:
            detailed_data = info['detailed_info']
            click.echo(f"\n📈 Recent M&A Activity:")
            for event in detailed_data.get('recent_ma_events', [])[:3]:
                click.echo(f"  - {event}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error getting company information: {e}")
        raise click.ClickException(str(e))


# Skill Shortage Analysis Commands

@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to analyze (can specify multiple)')
@click.option('--limit',
              '-l',
              type=int,
              help='Limit number of companies to analyze')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def skill_shortage_pipeline(years, limit, output):
    """Run complete skill shortage analysis pipeline"""

    if years:
        click.echo(f"🔍 Running skill shortage analysis for years: {list(years)}")
    else:
        click.echo("🔍 Running skill shortage analysis for all available years...")

    if limit:
        click.echo(f"Analysis limit: {limit} companies")

    try:
        system = FinancialVectorRAG()

        results = system.run_skill_shortage_analysis_pipeline(
            years=list(years) if years else [2022, 2023],
            limit_companies=limit)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Skill shortage analysis failed")

        # Display results summary
        click.echo(f"\n📊 Skill Shortage Analysis Results:")
        click.echo(f"Companies analyzed: {results.get('companies_analyzed', 0)}")
        click.echo(f"Filings analyzed: {results.get('filings_analyzed', 0)}")
        click.echo(f"Skill shortage findings: {results.get('skill_shortage_findings', 0)}")

        if results.get('output_file'):
            click.echo(f"📄 Results saved to: {results['output_file']}")

        if results.get('errors'):
            click.echo(f"\n⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                click.echo(f"  - {error}")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Analysis saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during skill shortage analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('company_ticker')
@click.option('--focus',
              '-f',
              help='Focus area for analysis (e.g., "technology", "healthcare")')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
@click.option('--format',
              type=click.Choice(['json', 'csv', 'txt']),
              default='json',
              help='Output format')
def skill_shortage_company(company_ticker, focus, output, format):
    """Analyze skill shortage for a specific company"""

    click.echo(f"🔍 Analyzing skill shortage for {company_ticker.upper()}...")
    if focus:
        click.echo(f"Focus area: {focus}")

    try:
        system = FinancialVectorRAG()

        results = system.analyze_company_skill_shortage(
            company_ticker=company_ticker.upper(),
            analysis_focus=focus if focus else "comprehensive skill shortage analysis")

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Company skill shortage analysis failed")

        # Display results
        click.echo(f"\n📊 Skill Shortage Analysis for {company_ticker.upper()}:")
        if 'company_info' in results:
            company_info = results['company_info']
            click.echo(f"Company: {company_info.get('name', 'Unknown')} ({company_info.get('ticker', 'N/A')})")
            click.echo(f"Sector: {company_info.get('sector', 'Unknown')}")

        # Get the correct document count
        documents_analyzed = results.get('chunk_count', results.get('total_chunks_found', 0))
        click.echo(f"Documents analyzed: {documents_analyzed}")
        
        # Get skill shortage data
        skill_shortage_data = results.get('skill_shortage_data', {})
        skill_shortage_mentions = skill_shortage_data.get('total_mentions', 0)
        likelihood_score = skill_shortage_data.get('average_likelihood_score', 0)
        recent_summary = skill_shortage_data.get('recent_summary', '')
        
        click.echo(f"Skill shortage mentions: {skill_shortage_mentions}")
        click.echo(f"AI Likelihood Score: {likelihood_score:.1f}/10")
        
        # Determine risk level
        if likelihood_score >= 7:
            risk_level = "High Risk"
        elif likelihood_score >= 5:
            risk_level = "Medium Risk"
        elif likelihood_score >= 3:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"
        click.echo(f"Risk Level: {risk_level}")
        
        # Display AI summary if available
        if recent_summary:
            click.echo(f"\n🤖 AI Summary:")
            # Show first few lines of the summary
            summary_lines = recent_summary.split('\n')[:3]
            for line in summary_lines:
                if line.strip():
                    click.echo(f"  {line.strip()}")
            if len(recent_summary.split('\n')) > 3:
                click.echo("  ...")

        # Display specific skill gaps if available in the analysis
        if 'skill_shortage_analysis' in results:
            click.echo(f"\n📊 Detailed Analysis:")
            # Show first few lines of the analysis
            analysis_lines = results['skill_shortage_analysis'].split('\n')[:5]
            for line in analysis_lines:
                if line.strip():
                    click.echo(f"  {line.strip()}")
        elif 'analysis' in results:
            click.echo(f"\n📊 General Analysis:")
            analysis_lines = results['analysis'].split('\n')[:5]
            for line in analysis_lines:
                if line.strip():
                    click.echo(f"  {line.strip()}")
        
        # Display document references if available
        if 'document_references' in results:
            doc_refs = results['document_references']
            click.echo(f"\n📄 Document References:")
            click.echo(f"Total chunks analyzed: {doc_refs.get('total_chunks_analyzed', 0)}")
            
            doc_summary = doc_refs.get('document_summary', {})
            if doc_summary and not doc_summary.get('error'):
                click.echo(f"Unique source documents: {doc_summary.get('unique_documents', 0)}")
                
                # Show date range
                date_range = doc_summary.get('date_range', {})
                if date_range.get('earliest') != 'Unknown' and date_range.get('latest') != 'Unknown':
                    click.echo(f"Filing date range: {date_range['earliest']} to {date_range['latest']}")
                
                # Show top source documents
                documents = doc_summary.get('documents', [])
                if documents:
                    click.echo(f"\n📋 Top Source Documents:")
                    for i, doc in enumerate(documents[:3], 1):  # Show top 3
                        click.echo(f"  {i}. {doc['document_id']}")
                        click.echo(f"     Form: {doc['form_type']}, Date: {doc['filing_date']}")
                        click.echo(f"     Chunks used: {doc['total_chunks']}, Avg similarity: {doc['avg_similarity']:.3f}")
                
                # Show highest similarity chunk info
                highest_chunk = doc_summary.get('highest_similarity_chunk')
                if highest_chunk:
                    click.echo(f"\n🎯 Highest similarity chunk: {highest_chunk['similarity_score']:.3f}")
            
            # Show source documents summary
            source_docs = doc_refs.get('source_documents', [])
            if source_docs:
                click.echo(f"\n🔍 Source Evidence Summary:")
                click.echo(f"Chunks used in analysis: {len(source_docs)}")
                
                # Show top 3 chunks by similarity
                sorted_chunks = sorted(source_docs, key=lambda x: x['similarity_score'], reverse=True)
                click.echo(f"\nTop evidence chunks:")
                for i, chunk in enumerate(sorted_chunks[:3], 1):
                    click.echo(f"  {i}. Chunk {chunk['chunk_index']} (Similarity: {chunk['similarity_score']:.3f})")
                    click.echo(f"     Document: {chunk['original_document_id']}")
                    click.echo(f"     Form: {chunk['form_type']}, Date: {chunk['filing_date']}")
                    # Show content preview
                    content_preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
                    click.echo(f"     Preview: {content_preview}")
                    click.echo("")
        
        # Show note if no skill shortage data
        if skill_shortage_data and skill_shortage_data.get('error'):
            click.echo(f"\nℹ️  {skill_shortage_data['error']}")
            click.echo("💡 To get detailed skill shortage statistics, run the skill shortage analysis pipeline first.")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, format)
            if success:
                click.echo(f"\n💾 Analysis saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during company skill shortage analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--companies',
              '-c',
              multiple=True,
              help='Specific companies to compare (can specify multiple)')
@click.option('--sector',
              '-s',
              help='Compare companies within a specific sector')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def skill_shortage_compare(companies, sector, output):
    """Compare skill shortage patterns across companies or sectors"""

    if companies:
        click.echo(f"🔍 Comparing skill shortage patterns for: {list(companies)}")
    elif sector:
        click.echo(f"🔍 Comparing skill shortage patterns in {sector} sector")
    else:
        click.echo("🔍 Comparing skill shortage patterns across all companies")

    try:
        system = FinancialVectorRAG()

        results = system.compare_skill_shortage_across_companies(
            company_tickers=list(companies) if companies else None,
            sector=sector)

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Skill shortage comparison failed")

        # Display results
        click.echo(f"\n📊 Skill Shortage Comparison Results:")
        click.echo(f"Companies compared: {results.get('companies_compared', 0)}")
        click.echo(f"Average severity score: {results.get('avg_severity_score', 0):.2f}/10")

        # Display comparison data
        if 'comparison_data' in results:
            click.echo(f"\n📈 Company Rankings (by severity):")
            for i, company_data in enumerate(results['comparison_data'][:10], 1):
                click.echo(f"  {i}. {company_data['company']}: {company_data['severity_score']:.2f}")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Comparison saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during skill shortage comparison: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to analyze (can specify multiple)')
@click.option('--sector', '-s', help='Focus on specific sector')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def skill_shortage_trends(years, sector, output):
    """Analyze skill shortage trends over time and across sectors"""

    if years:
        click.echo(f"📈 Analyzing skill shortage trends for years: {list(years)}")
    else:
        click.echo("📈 Analyzing skill shortage trends across all available years...")
    
    if sector:
        click.echo(f"Focusing on {sector} sector")

    try:
        system = FinancialVectorRAG()

        results = system.analyze_skill_shortage_trends(
            years=list(years) if years else None,
            sector=sector
        )

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("Trends analysis failed")

        # Display results summary
        click.echo(f"\n📊 Trends Analysis Results:")
        click.echo(f"Data points: {results.get('data_points', 0)}")
        click.echo(f"Years covered: {results.get('years_covered', [])}")
        click.echo(f"Companies covered: {results.get('companies_covered', 0)}")

        # Display trend data
        if 'trend_analysis' in results:
            trend_data = results['trend_analysis']
            if 'yearly_trends' in trend_data:
                click.echo("\n📅 Yearly Trends:")
                for year, data in sorted(trend_data['yearly_trends'].items()):
                    click.echo(f"  {year}: {data['total_mentions']} mentions, "
                             f"{data['mention_rate']:.1%} mention rate")

        # Save to file if requested
        if output:
            success = system.export_analysis_report(results, output, 'json')
            if success:
                click.echo(f"\n💾 Analysis saved to: {output}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during trends analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.option('--limit',
              '-l',
              type=int,
              help='Limit number of filings to process')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def analyze_csv(csv_path, limit, output):
    """Analyze skill shortage from CSV data (similar to original provided code)"""

    click.echo(f"📊 Analyzing skill shortage from CSV: {csv_path}")
    if limit:
        click.echo(f"Processing limit: {limit} filings")

    try:
        system = FinancialVectorRAG()

        results = system.analyze_from_csv_data(
            csv_path=csv_path,
            limit=limit
        )

        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            raise click.ClickException("CSV analysis failed")

        # Display results
        click.echo(f"\n✅ CSV Analysis completed!")
        click.echo(f"📊 Analysis results: {results['analysis_results']}")
        click.echo(f"🎯 Significant findings: {results['significant_findings']}")
        
        if 'output_file' in results:
            click.echo(f"💾 Results saved to: {results['output_file']}")

        # Display summary stats
        if 'summary_stats' in results:
            stats = results['summary_stats']
            click.echo(f"\n📊 Summary Statistics:")
            click.echo(f"Total filings analyzed: {stats.get('total_filings_analyzed', 0)}")
            click.echo(f"Filings with mentions: {stats.get('filings_with_skill_shortage_mentions', 0)}")
            click.echo(f"Mention rate: {stats.get('skill_shortage_mention_rate', 0):.1%}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error during CSV analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
def skill_shortage_stats():
    """Get summary statistics about skill shortage analysis"""

    click.echo("📊 Skill Shortage Analysis Statistics")

    try:
        system = FinancialVectorRAG()

        stats = system.get_skill_shortage_statistics()

        if 'error' in stats:
            click.echo(f"❌ {stats['error']}")
            raise click.ClickException("Failed to get skill shortage statistics")

        # Display statistics
        click.echo(f"\n📈 Skill Shortage Data Statistics:")
        click.echo(f"Total companies analyzed: {stats.get('total_companies_analyzed', 0)}")
        click.echo(f"Total skill shortage mentions: {stats.get('total_skill_shortage_mentions', 0)}")
        click.echo(f"Average severity score: {stats.get('avg_severity_score', 0):.2f}/10")
        click.echo(f"Most affected sectors: {stats.get('most_affected_sectors', [])}")

        # Display recent findings
        if 'recent_findings' in stats:
            click.echo(f"\n🕒 Recent Findings:")
            for finding in stats['recent_findings'][:5]:
                click.echo(f"  - {finding}")

        system.close()

    except Exception as e:
        click.echo(f"❌ Error getting skill shortage statistics: {e}")
        raise click.ClickException(str(e))


# Hiring Difficulties Analysis Commands

@cli.command()
@click.option('--years',
              '-y',
              multiple=True,
              type=int,
              help='Specific years to analyze (can specify multiple)')
@click.option('--limit',
              '-l',
              type=int,
              help='Limit number of companies to analyze')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def hiring_difficulties_pipeline(years, limit, output):
    """Run the complete hiring difficulties analysis pipeline"""
    
    click.echo("🚀 Starting Hiring Difficulties Analysis Pipeline")
    
    # Default to recent years if none specified
    if not years:
        years = [2022, 2023]
        click.echo(f"📅 Using default years: {list(years)}")
    else:
        years = list(years)
        click.echo(f"📅 Analyzing years: {years}")
    
    if limit:
        click.echo(f"🏢 Limiting to {limit} companies")
    
    try:
        system = FinancialVectorRAG()
        
        # Run the pipeline
        results = system.run_hiring_difficulties_analysis_pipeline(
            years=years,
            limit_companies=limit,
            save_results=True
        )
        
        if 'errors' in results and results['errors']:
            click.echo("⚠️ Pipeline completed with errors:")
            for error in results['errors']:
                click.echo(f"  - {error}")
        
        # Display results
        click.echo(f"\n✅ Pipeline completed successfully!")
        click.echo(f"Companies analyzed: {results.get('companies_analyzed', 0)}")
        click.echo(f"Filings analyzed: {results.get('filings_analyzed', 0)}")
        click.echo(f"Hiring difficulty findings: {results.get('hiring_difficulty_findings', 0)}")
        
        if 'output_file' in results:
            click.echo(f"📄 Results saved to: {results['output_file']}")
        
        # Save to custom output if specified
        if output:
            rankings = system.rank_companies_by_hiring_difficulties()
            export_path = system.export_hiring_difficulties_rankings(
                rankings=rankings,
                output_path=output
            )
            click.echo(f"📊 Rankings exported to: {export_path}")
        
        system.close()
        
    except Exception as e:
        click.echo(f"❌ Error during hiring difficulties analysis: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('company_ticker')
@click.option('--focus',
              '-f',
              help='Focus area for analysis (e.g., "technical skills", "labor shortage")')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
@click.option('--format',
              type=click.Choice(['json', 'csv', 'txt']),
              default='json',
              help='Output format')
def hiring_difficulties_company(company_ticker, focus, output, format):
    """Analyze hiring difficulties for a specific company"""
    
    company_ticker = company_ticker.upper()
    click.echo(f"🔍 Analyzing hiring difficulties for {company_ticker}")
    
    if focus:
        click.echo(f"🎯 Focus area: {focus}")
    
    try:
        system = FinancialVectorRAG()
        
        # Get analysis focus
        analysis_focus = focus if focus else "comprehensive hiring difficulties analysis"
        
        # Run analysis
        results = system.analyze_company_hiring_difficulties(
            company_ticker=company_ticker,
            analysis_focus=analysis_focus
        )
        
        if 'error' in results:
            click.echo(f"❌ {results['error']}")
            if 'available_companies' in results:
                click.echo("Available companies:")
                for ticker in results['available_companies'][:10]:
                    click.echo(f"  - {ticker}")
            raise click.ClickException("Company analysis failed")
        
        # Display results
        click.echo(f"\n📊 Hiring Difficulties Analysis for {company_ticker}")
        
        if 'company_info' in results:
            info = results['company_info']
            click.echo(f"Company: {info.get('name', 'Unknown')}")
            click.echo(f"Sector: {info.get('sector', 'Unknown')}")
            click.echo(f"Headquarters: {info.get('headquarters', 'Unknown')}")
        
        if 'hiring_difficulty_score' in results:
            score = results['hiring_difficulty_score']
            click.echo(f"\n🎯 Hiring Difficulty Score: {score:.2f}")
        
        if 'hiring_difficulty_likelihood' in results:
            likelihood = results['hiring_difficulty_likelihood']
            click.echo(f"📈 Hiring Difficulty Likelihood: {likelihood:.1%}")
        
        if 'key_findings' in results:
            click.echo(f"\n🔍 Key Findings:")
            for finding in results['key_findings'][:5]:
                click.echo(f"  • {finding}")
        
        if 'recent_mentions' in results:
            click.echo(f"\n📅 Recent Mentions: {results['recent_mentions']}")
        
        # Save results if requested
        if output:
            success = system.export_analysis_report(results, output, format)
            if success:
                click.echo(f"\n💾 Results saved to: {output}")
        
        system.close()
        
    except Exception as e:
        click.echo(f"❌ Error analyzing company hiring difficulties: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--min-filings',
              '-m',
              type=int,
              default=1,
              help='Minimum number of filings required for ranking')
@click.option('--recent-only',
              '-r',
              is_flag=True,
              help='Only consider recent filings')
@click.option('--years-lookback',
              '-y',
              type=int,
              default=3,
              help='Number of recent years to consider if --recent-only is used')
@click.option('--output',
              '-o',
              type=click.Path(),
              help='Output file path for results')
def hiring_difficulties_rankings(min_filings, recent_only, years_lookback, output):
    """Rank companies by hiring difficulties"""
    
    click.echo("📊 Ranking companies by hiring difficulties")
    
    if recent_only:
        click.echo(f"🕒 Considering only filings from last {years_lookback} years")
    
    click.echo(f"📋 Minimum filings required: {min_filings}")
    
    try:
        system = FinancialVectorRAG()
        
        # Get rankings
        rankings = system.rank_companies_by_hiring_difficulties(
            min_filings=min_filings,
            recent_years_only=recent_only,
            years_lookback=years_lookback
        )
        
        if not rankings:
            click.echo("❌ No rankings available. Run the hiring difficulties pipeline first.")
            raise click.ClickException("No data available for ranking")
        
        # Display top 20 rankings
        click.echo(f"\n🏆 Top Companies by Hiring Difficulties:")
        click.echo(f"{'Rank':<4} {'Company':<30} {'Score':<8} {'Likelihood':<12} {'Filings':<8}")
        click.echo("-" * 70)
        
        for i, company in enumerate(rankings[:20], 1):
            name = company.get('company_name', 'Unknown')[:28]
            score = company.get('avg_hiring_difficulty_score', 0)
            likelihood = company.get('avg_hiring_difficulty_likelihood', 0)
            filings = company.get('filing_count', 0)
            
            click.echo(f"{i:<4} {name:<30} {score:<8.2f} {likelihood:<12.1%} {filings:<8}")
        
        # Save results if requested
        if output:
            export_path = system.export_hiring_difficulties_rankings(
                rankings=rankings,
                output_path=output
            )
            click.echo(f"\n💾 Full rankings saved to: {export_path}")
        
        system.close()
        
    except Exception as e:
        click.echo(f"❌ Error generating hiring difficulties rankings: {e}")
        raise click.ClickException(str(e))


@cli.command()
def hiring_difficulties_stats():
    """Show hiring difficulties analysis statistics"""
    
    click.echo("📊 Hiring Difficulties Analysis Statistics")
    
    try:
        system = FinancialVectorRAG()
        
        stats = system.get_hiring_difficulties_summary_stats()
        
        if 'error' in stats:
            click.echo(f"❌ {stats['error']}")
            raise click.ClickException("Failed to get statistics")
        
        # Display statistics
        click.echo(f"\n📈 Analysis Coverage:")
        click.echo(f"Total companies analyzed: {stats.get('total_companies', 0)}")
        click.echo(f"Total filings processed: {stats.get('total_filings', 0)}")
        click.echo(f"Companies with hiring difficulties: {stats.get('companies_with_difficulties', 0)}")
        click.echo(f"Significant findings: {stats.get('significant_findings', 0)}")
        
        if 'avg_hiring_difficulty_score' in stats:
            click.echo(f"\n🎯 Average Scores:")
            click.echo(f"Average hiring difficulty score: {stats['avg_hiring_difficulty_score']:.2f}")
            click.echo(f"Average hiring difficulty likelihood: {stats['avg_hiring_difficulty_likelihood']:.1%}")
        
        if 'top_difficulty_terms' in stats:
            click.echo(f"\n🔍 Most Common Hiring Difficulty Indicators:")
            for term, count in stats['top_difficulty_terms'][:10]:
                click.echo(f"  • {term}: {count} mentions")
        
        if 'years_covered' in stats:
            click.echo(f"\n📅 Years covered: {', '.join(map(str, stats['years_covered']))}")
        
        system.close()
        
    except Exception as e:
        click.echo(f"❌ Error getting hiring difficulties statistics: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    cli()