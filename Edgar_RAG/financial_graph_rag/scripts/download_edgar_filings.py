#!/usr/bin/env python3
"""
Standalone EDGAR 10-K Filing Downloader Script

This script downloads 10-K filings from EDGAR for S&P 500 companies
and optionally processes them for M&A content analysis.

Usage:
    python download_edgar_filings.py --help
    python download_edgar_filings.py --tickers AAPL MSFT --years 2023
    python download_edgar_filings.py --limit-companies 5 --years 2022 2023
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

# Add parent directory to path to import financial_graph_rag modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial_graph_rag.data_collectors import SP500Collector, EdgarFilingCollector
from financial_graph_rag.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_filings_for_companies(companies: List,
                                   years: List[int],
                                   limit_per_company: Optional[int] = None,
                                   ma_only: bool = False,
                                   output_dir: Optional[str] = None) -> dict:
    """
    Download 10-K filings for the specified companies and years
    
    Args:
        companies: List of SP500Company objects
        years: List of years to download
        limit_per_company: Maximum filings per company
        ma_only: Only return M&A relevant filings
        output_dir: Custom output directory
        
    Returns:
        Dictionary with download results and statistics
    """
    logger.info(
        f"Starting download for {len(companies)} companies, years: {years}")

    # Initialize collector
    edgar_collector = EdgarFilingCollector()

    if output_dir:
        edgar_collector.data_directory = Path(output_dir)
        edgar_collector.data_directory.mkdir(parents=True, exist_ok=True)

    # Download filings
    all_filings = []
    successful_companies = 0
    failed_companies = 0

    for i, company in enumerate(companies, 1):
        if not company.cik:
            logger.warning(f"Skipping {company.symbol} - no CIK available")
            continue

        logger.info(f"Processing {company.symbol} ({i}/{len(companies)})")

        try:
            company_filings = edgar_collector._download_company_filings(
                company, years, limit_per_company)

            if company_filings:
                all_filings.extend(company_filings)
                successful_companies += 1
                logger.info(
                    f"  ✅ Downloaded {len(company_filings)} filings for {company.symbol}"
                )
            else:
                logger.warning(f"  ⚠️  No filings found for {company.symbol}")

        except Exception as e:
            logger.error(
                f"  ❌ Error downloading filings for {company.symbol}: {e}")
            failed_companies += 1
            continue

    # Update collector's filings list
    edgar_collector.filings = all_filings

    # Filter for M&A content if requested
    if ma_only:
        ma_filings = edgar_collector.filter_ma_relevant_filings()
        logger.info(
            f"Filtered to {len(ma_filings)} M&A relevant filings (from {len(all_filings)} total)"
        )
        relevant_filings = ma_filings
    else:
        relevant_filings = all_filings

    # Save cache
    edgar_collector.save_cache()

    # Generate statistics
    stats = edgar_collector.get_summary_stats()
    stats.update({
        'successful_companies': successful_companies,
        'failed_companies': failed_companies,
        'relevant_filings_count': len(relevant_filings)
    })

    return {
        'filings': relevant_filings,
        'stats': stats,
        'collector': edgar_collector
    }


def save_results_report(results: dict, output_file: str):
    """Save download results to a JSON report file"""
    try:
        report = {'download_summary': results['stats'], 'filings': []}

        # Add filing details (without full text content to keep file manageable)
        for filing in results['filings']:
            filing_info = {
                'cik': filing.cik,
                'company_name': filing.company_name,
                'form_type': filing.form_type,
                'filing_date': filing.filing_date,
                'accession_number': filing.accession_number,
                'ma_score': filing.ma_score,
                'has_ma_content': filing.has_ma_content,
                'processed': filing.processed,
                'error': filing.error
            }
            report['filings'].append(filing_info)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved download report to {output_file}")

    except Exception as e:
        logger.error(f"Error saving report: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Download 10-K filings from EDGAR for S&P 500 companies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2023 filings for Apple and Microsoft
  python download_edgar_filings.py --tickers AAPL MSFT --years 2023
  
  # Download M&A relevant filings for top 10 companies
  python download_edgar_filings.py --limit-companies 10 --years 2022 2023 --ma-only
  
  # Download recent filings to custom directory
  python download_edgar_filings.py --years 2023 --limit-companies 5 --output-dir ./my_filings
        """)

    parser.add_argument('--years',
                        '-y',
                        type=int,
                        nargs='+',
                        default=[2023],
                        help='Years of filings to download (default: 2023)')

    parser.add_argument(
        '--tickers',
        '-t',
        nargs='+',
        help='Specific tickers to download (e.g., AAPL MSFT GOOGL)')

    parser.add_argument('--limit-companies',
                        '-c',
                        type=int,
                        help='Limit number of companies to process')

    parser.add_argument('--limit-filings',
                        '-f',
                        type=int,
                        default=1,
                        help='Limit filings per company (default: 1)')

    parser.add_argument('--ma-only',
                        action='store_true',
                        help='Only download/keep filings with M&A content')

    parser.add_argument('--output-dir',
                        '-o',
                        help='Custom output directory for downloaded files')

    parser.add_argument('--report-file',
                        '-r',
                        help='Save download report to JSON file')

    parser.add_argument('--verbose',
                        '-v',
                        action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize S&P 500 collector
        sp500_collector = SP500Collector()

        # Get companies to process
        if args.tickers:
            # Get specific companies by ticker
            companies = []
            for ticker in args.tickers:
                company = sp500_collector.get_company_by_symbol(ticker.upper())
                if company:
                    companies.append(company)
                    print(f"✅ Found {ticker.upper()}: {company.name}")
                else:
                    print(f"⚠️  Ticker {ticker.upper()} not found in S&P 500")

            if not companies:
                print("❌ No valid companies found")
                return 1
        else:
            # Get all companies with CIK
            companies = sp500_collector.get_companies_with_cik()
            if args.limit_companies:
                companies = companies[:args.limit_companies]

        print(f"📊 Processing {len(companies)} companies")
        print(f"📅 Years: {args.years}")
        print(f"📄 Filings per company: {args.limit_filings}")
        if args.ma_only:
            print("🎯 M&A relevant filings only")
        if args.output_dir:
            print(f"📁 Output directory: {args.output_dir}")

        # Download filings
        results = download_filings_for_companies(
            companies=companies,
            years=args.years,
            limit_per_company=args.limit_filings,
            ma_only=args.ma_only,
            output_dir=args.output_dir)

        # Display results
        stats = results['stats']
        print("\n✅ Download completed!")
        print(f"📊 Total filings downloaded: {stats['total_filings']}")
        print(f"📊 Successfully processed: {stats['processed_filings']}")
        print(f"📊 M&A relevant filings: {stats['ma_relevant_filings']}")
        print(f"📊 Companies successful: {stats['successful_companies']}")
        print(f"📊 Companies failed: {stats['failed_companies']}")

        if stats['ma_relevant_filings'] > 0:
            ma_rate = stats['ma_relevance_rate'] * 100
            print(f"📊 M&A relevance rate: {ma_rate:.1f}%")

        # Show top companies by filing count
        if stats['company_breakdown']:
            print("\n📈 Top companies by filing count:")
            sorted_companies = sorted(stats['company_breakdown'].items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:5]
            for company, count in sorted_companies:
                print(f"  • {company}: {count} filings")

        # Show storage location
        print(f"\n💾 Files stored in: {results['collector'].data_directory}")

        # Save report if requested
        if args.report_file:
            save_results_report(results, args.report_file)

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during download: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
