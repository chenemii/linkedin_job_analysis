"""
Streamlit UI for Financial Vector RAG System

A comprehensive web interface for viewing vector store contents and querying
the Financial Vector RAG system for M&A analysis and skill shortage analysis.
"""

# Fix tokenizers fork warning - must be set before any imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from pathlib import Path

# Import the Financial Vector RAG system
from financial_graph_rag.core import FinancialVectorRAG
from financial_graph_rag.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Financial Graph RAG Explorer",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .company-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .query-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .skill-shortage-result {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .document-content {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .query-config {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .skill-gap-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 0.5rem;
    }
</style>
""",
            unsafe_allow_html=True)

# Global system instance to avoid reinitialization
@st.cache_resource(show_spinner=False)
def get_system_instance():
    """Get a cached system instance to avoid reinitialization"""
    try:
        logger.info("Initializing Financial Vector RAG system (cached)")
        system = FinancialVectorRAG()
        logger.info("Financial Vector RAG system initialized successfully")
        return system
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return None

# Use the cached system instance
def initialize_system():
    """Get the cached system instance"""
    return get_system_instance()

# Cache expensive operations with longer TTL
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_system_status():
    """Get system status information"""
    system = initialize_system()
    if system:
        try:
            status = system.get_system_status()
            return status
        except Exception as e:
            st.error(f"Error getting system status: {e}")
            return None
    return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_companies():
    """Get list of available companies with documents"""
    system = initialize_system()
    if system:
        try:
            companies = system.rag_engine.get_available_companies()
            # Filter out companies with 0 documents
            companies_with_data = [
                c for c in companies if c.get('document_count', 0) > 0
            ]
            return companies_with_data
        except Exception as e:
            st.error(f"Error getting companies: {e}")
            return []
    return []

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_sp500_companies():
    """Get S&P 500 companies with sector information, filtered to only those with data"""
    system = initialize_system()
    if system:
        try:
            # Get all S&P 500 companies
            all_sp500_companies = system.sp500_collector.collect_all()

            # Get companies that have data in vector store
            companies_with_data = get_available_companies()
            companies_with_data_tickers = {
                c['company_ticker']
                for c in companies_with_data
            }

            # Filter S&P 500 companies to only those with data
            sp500_with_data = [
                c for c in all_sp500_companies
                if c.symbol in companies_with_data_tickers
            ]

            return sp500_with_data
        except Exception as e:
            st.error(f"Error getting S&P 500 companies: {e}")
            return []
    return []

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_sectors():
    """Get list of available sectors from companies that have data"""
    system = initialize_system()
    if system:
        try:
            # Get only companies with data
            sp500_companies_with_data = get_sp500_companies()
            sectors = list(
                set(c.sector for c in sp500_companies_with_data if c.sector))
            return sorted(sectors)
        except Exception as e:
            st.error(f"Error getting sectors: {e}")
            return []
    return []

@st.cache_data(ttl=300)  # Cache for 5 minutes (shorter since this can change)
def check_skill_shortage_analysis_availability():
    """Check if skill shortage analysis results are available"""
    system = initialize_system()
    if system and hasattr(system, 'skill_shortage_analyzer') and system.skill_shortage_analyzer:
        try:
            # First check if results are already in memory
            if system.skill_shortage_analyzer.results:
                return {
                    'available': True,
                    'count': len(system.skill_shortage_analyzer.results),
                    'source': 'memory'
                }
            
            # Try to load from cache
            if system.skill_shortage_analyzer.load_cache():
                count = len(system.skill_shortage_analyzer.results)
                # Clear results from memory to avoid keeping them loaded
                system.skill_shortage_analyzer.results = []
                return {
                    'available': True,
                    'count': count,
                    'source': 'cache'
                }
            
            return {
                'available': False,
                'count': 0,
                'source': None
            }
        except Exception as e:
            logger.error(f"Error checking skill shortage analysis availability: {e}")
            return {
                'available': False,
                'count': 0,
                'source': None,
                'error': str(e)
            }
    return {
        'available': False,
        'count': 0,
        'source': None,
        'error': 'Skill shortage analyzer not available'
    }

def format_company_data(companies: List[Dict]) -> pd.DataFrame:
    """Format company data for display"""
    if not companies:
        return pd.DataFrame()

    df = pd.DataFrame(companies)

    # Clean up the data
    df['Company Ticker'] = df['company_ticker']
    df['Company Name'] = df['company_name']
    df['Document Count'] = df['document_count']
    df['Collection Name'] = df['collection_name']
    df['Created At'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Select and reorder columns
    display_columns = [
        'Company Ticker', 'Company Name', 'Document Count', 'Created At'
    ]
    df = df[display_columns].copy()

    return df


def display_company_overview():
    """Display company overview page"""
    st.markdown('<h1 class="main-header">📊 Vector Store Overview</h1>',
                unsafe_allow_html=True)

    # Get system status
    status = get_system_status()
    companies = get_available_companies()

    if not status:
        st.error(
            "Unable to connect to the system. Please check your configuration."
        )
        return

    # Display system metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Companies", len(companies))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        total_docs = sum(c.get('document_count', 0) for c in companies)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Documents", total_docs)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        vector_status = status.get('vector_store', {})
        collections = vector_status.get('company_collections', 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Collections", collections)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        sp500_status = status.get('sp500_data', {})
        sp500_companies = sp500_status.get('companies_loaded', 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("S&P 500 Companies", sp500_companies)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Display companies table
    st.subheader("📋 Companies in Vector Store")

    if companies:
        df = format_company_data(companies)

        # Configure AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_selection('single', use_checkbox=True)
        gb.configure_default_column(groupable=True,
                                    value=True,
                                    enableRowGroup=True,
                                    aggFunc='sum',
                                    editable=False)

        grid_options = gb.build()

        # Display grid
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            theme='streamlit',
            height=400)

        # Show selected company details
        if grid_response['selected_rows'] is not None and len(
                grid_response['selected_rows']) > 0:
            selected_company = grid_response['selected_rows'][0]
            ticker = selected_company['Company Ticker']

            st.subheader(f"📈 Details for {ticker}")
            display_company_details(ticker)
    else:
        st.warning(
            "No companies found in the vector store. Please run the setup pipeline first."
        )

    # Display charts
    if companies:
        st.markdown("---")
        st.subheader("📊 Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Document count by company
            df_chart = pd.DataFrame(companies)
            fig = px.bar(df_chart.head(10),
                         x='company_ticker',
                         y='document_count',
                         title="Document Count by Company (Top 10)",
                         labels={
                             'company_ticker': 'Company',
                             'document_count': 'Documents'
                         })
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Document distribution pie chart
            total_docs = sum(c.get('document_count', 0) for c in companies)
            if total_docs > 0:
                # Group smaller companies
                df_pie = pd.DataFrame(companies)
                df_pie = df_pie.sort_values('document_count', ascending=False)

                if len(df_pie) > 8:
                    top_companies = df_pie.head(7)
                    others_count = df_pie.tail(len(df_pie) -
                                               7)['document_count'].sum()
                    others_row = pd.DataFrame([{
                        'company_ticker': 'Others',
                        'document_count': others_count
                    }])
                    df_pie = pd.concat([top_companies, others_row],
                                       ignore_index=True)

                fig = px.pie(df_pie,
                             values='document_count',
                             names='company_ticker',
                             title="Document Distribution")
                st.plotly_chart(fig, use_container_width=True)


def display_company_details(ticker: str):
    """Display detailed information for a specific company"""
    system = initialize_system()
    if not system:
        return

    try:
        # Get company statistics
        stats = system.rag_engine.get_company_statistics(ticker)

        if stats.get('status') == 'error':
            st.error(
                f"Error getting details for {ticker}: {stats.get('error')}")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            st.write(f"**Company:** {ticker}")
            st.write(f"**Document Count:** {stats.get('document_count', 0)}")
            st.write(f"**Status:** {stats.get('status', 'Unknown')}")

            avg_score = stats.get('avg_ma_score')
            if avg_score is not None:
                st.write(f"**Average M&A Score:** {avg_score:.3f}")

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            form_types = stats.get('form_types', [])
            if form_types:
                st.write(f"**Form Types:** {', '.join(form_types)}")

            filing_dates = stats.get('sample_filing_dates', [])
            if filing_dates:
                st.write(f"**Recent Filing Dates:**")
                for date in filing_dates[:3]:
                    st.write(f"  - {date}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Sample documents
        if stats.get('document_count', 0) > 0:
            st.subheader("📄 Sample Documents")

            # Get sample documents
            sample_docs = system.rag_engine.retrieve_company_documents(
                query="merger acquisition organizational",
                company_ticker=ticker,
                top_k=3)

            for i, doc in enumerate(sample_docs):
                with st.expander(
                        f"Document {i+1} (Score: {doc['score']:.3f})"):
                    st.markdown('<div class="document-content">',
                                unsafe_allow_html=True)
                    content = doc['content']
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    st.text(content)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show metadata
                    if 'metadata' in doc and doc['metadata']:
                        st.json(doc['metadata'])

    except Exception as e:
        st.error(f"Error getting company details: {e}")


def display_query_interface():
    """Display the enhanced query interface"""
    st.markdown('<h1 class="main-header">🔍 Enhanced Query Interface</h1>',
                unsafe_allow_html=True)

    system = initialize_system()
    if not system:
        st.error("Unable to connect to the system.")
        return

    companies = get_available_companies()
    company_tickers = [c['company_ticker'] for c in companies]
    sectors = get_available_sectors()

    # Query configuration section
    st.markdown('<div class="query-config">', unsafe_allow_html=True)
    st.subheader("🎯 Query Configuration")

    # Query input
    query = st.text_area(
        "Enter your M&A analysis query:",
        placeholder=
        "e.g., cultural integration merger acquisition, organizational restructuring, synergies cost savings",
        height=100)

    # Query type selection
    col1, col2 = st.columns(2)

    with col1:
        query_type = st.selectbox("Query Scope:", [
            "Cross-Company Search", "Company-Specific Search",
            "Sector Analysis", "Custom Company Selection"
        ])

    with col2:
        use_ai_analysis = st.checkbox(
            "Use AI Analysis",
            value=True,
            help="Get LLM-powered insights vs raw documents")

    # Dynamic configuration based on query type
    selected_companies = []
    selected_sector = None

    if query_type == "Company-Specific Search":
        col1, col2 = st.columns(2)
        with col1:
            selected_company = st.selectbox(
                "Select Company:",
                options=[""] + sorted(company_tickers),
                help="Choose a specific company to analyze")
            if selected_company:
                selected_companies = [selected_company]

        with col2:
            # Show company info if selected
            if selected_company:
                sp500_companies = get_sp500_companies()
                company_info = next(
                    (c
                     for c in sp500_companies if c.symbol == selected_company),
                    None)
                if company_info:
                    st.info(
                        f"**{company_info.name}**\n\nSector: {company_info.sector}\n\nHeadquarters: {company_info.headquarters}"
                    )

    elif query_type == "Sector Analysis":
        col1, col2 = st.columns(2)
        with col1:
            selected_sector = st.selectbox(
                "Select Sector:",
                options=[""] + sectors,
                help="Analyze all companies within a specific sector")

        with col2:
            if selected_sector:
                # Get companies in selected sector that have data
                companies_with_data = get_available_companies()
                companies_with_data_tickers = {
                    c['company_ticker']
                    for c in companies_with_data
                }

                sp500_companies = get_sp500_companies()
                sector_companies = [
                    c.symbol for c in sp500_companies
                    if c.sector == selected_sector
                ]
                available_sector_companies = [
                    c for c in sector_companies
                    if c in companies_with_data_tickers
                ]
                selected_companies = available_sector_companies

                st.info(
                    f"**{selected_sector}**\n\n{len(available_sector_companies)} companies with data available\n\nCompanies: {', '.join(available_sector_companies[:5])}{'...' if len(available_sector_companies) > 5 else ''}"
                )

    elif query_type == "Custom Company Selection":
        st.write("Select multiple companies for comparison:")

        # Multi-select for companies
        selected_companies = st.multiselect(
            "Choose Companies:",
            options=sorted(company_tickers),
            help="Select multiple companies to compare")

        if selected_companies:
            st.info(
                f"Selected {len(selected_companies)} companies: {', '.join(selected_companies)}"
            )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            max_results = st.slider(
                "Max Results per Company",
                1,
                10,
                3,
                help="Number of documents to retrieve per company")

        with col2:
            if query_type in [
                    "Cross-Company Search", "Sector Analysis",
                    "Custom Company Selection"
            ]:
                max_companies = st.slider(
                    "Max Companies to Analyze",
                    1,
                    50,
                    20,
                    help="Limit the number of companies to analyze")
            else:
                max_companies = 1

        with col3:
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                help="Show document metadata in results")

    st.markdown('</div>', unsafe_allow_html=True)

    # Query execution
    if st.button("🚀 Execute Query", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        # Validate selections
        if query_type == "Company-Specific Search" and not selected_companies:
            st.warning("Please select a company for company-specific search.")
            return
        elif query_type == "Sector Analysis" and not selected_sector:
            st.warning("Please select a sector for sector analysis.")
            return
        elif query_type == "Custom Company Selection" and not selected_companies:
            st.warning(
                "Please select at least one company for custom selection.")
            return

        with st.spinner("Executing query..."):
            try:
                if query_type == "Company-Specific Search":
                    # Single company analysis
                    execute_single_company_query(system, query,
                                                 selected_companies[0],
                                                 use_ai_analysis, max_results,
                                                 include_metadata)

                elif query_type == "Sector Analysis":
                    # Sector-wide analysis
                    execute_sector_query(system, query, selected_sector,
                                         selected_companies, use_ai_analysis,
                                         max_results, max_companies,
                                         include_metadata)

                elif query_type == "Custom Company Selection":
                    # Custom company selection
                    execute_multi_company_query(system, query,
                                                selected_companies,
                                                use_ai_analysis, max_results,
                                                max_companies,
                                                include_metadata)

                else:  # Cross-Company Search
                    # All companies
                    execute_cross_company_query(system, query, company_tickers,
                                                use_ai_analysis, max_results,
                                                max_companies,
                                                include_metadata)

            except Exception as e:
                st.error(f"Error executing query: {e}")


def execute_single_company_query(system, query, company, use_ai_analysis,
                                 max_results, include_metadata):
    """Execute query for a single company"""
    st.subheader(f"📈 Results for {company}")

    if use_ai_analysis:
        results = system.rag_engine.analyze_company_ma_impact(
            query=query, company_ticker=company, top_k=max_results)
        display_company_analysis_results(results, query)
    else:
        documents = system.rag_engine.retrieve_company_documents(
            query=query, company_ticker=company, top_k=max_results)
        display_raw_documents(documents, query, company, include_metadata)


def execute_sector_query(system, query, sector, companies, use_ai_analysis,
                         max_results, max_companies, include_metadata):
    """Execute query for a specific sector"""
    st.subheader(f"🏭 Sector Analysis: {sector}")

    # Get companies in this sector that actually have data
    companies_with_data = get_available_companies()
    companies_with_data_tickers = {
        c['company_ticker']
        for c in companies_with_data
    }

    # Filter to only companies that have data
    sector_companies_with_data = [
        ticker for ticker in companies if ticker in companies_with_data_tickers
    ]

    if not sector_companies_with_data:
        st.warning(f"No companies with data found in {sector} sector.")
        return

    # Limit companies if needed
    if len(sector_companies_with_data) > max_companies:
        sector_companies_with_data = sector_companies_with_data[:max_companies]
        st.info(
            f"Analyzing top {max_companies} companies in {sector} sector with data"
        )

    if use_ai_analysis:
        # Use the sector trends analysis
        results = system.get_ma_trends_analysis(sector=sector)
        display_sector_analysis_results(results, query, sector)
    else:
        # Raw document search across sector companies
        results = system.rag_engine.search_across_companies(
            query=query,
            company_tickers=sector_companies_with_data,
            top_k_per_company=max_results)
        display_cross_company_raw_results(results, query, f"Sector: {sector}",
                                          include_metadata)


def execute_multi_company_query(system, query, companies, use_ai_analysis,
                                max_results, max_companies, include_metadata):
    """Execute query for custom selected companies"""
    st.subheader(f"🔍 Custom Company Analysis ({len(companies)} companies)")

    # Filter to only companies that have data
    companies_with_data = get_available_companies()
    companies_with_data_tickers = {
        c['company_ticker']
        for c in companies_with_data
    }

    companies_with_data_filtered = [
        c for c in companies if c in companies_with_data_tickers
    ]

    if not companies_with_data_filtered:
        st.warning("None of the selected companies have data available.")
        return

    if len(companies_with_data_filtered) < len(companies):
        excluded = set(companies) - set(companies_with_data_filtered)
        st.warning(f"Excluded companies with no data: {', '.join(excluded)}")

    # Limit companies if needed
    if len(companies_with_data_filtered) > max_companies:
        companies_with_data_filtered = companies_with_data_filtered[:
                                                                    max_companies]
        st.info(
            f"Analyzing first {max_companies} selected companies with data")

    st.info(
        f"Analyzing {len(companies_with_data_filtered)} companies with data: {', '.join(companies_with_data_filtered)}"
    )

    if use_ai_analysis:
        results = system.analyze_cross_company_query(
            query=query,
            company_tickers=companies_with_data_filtered,
            top_k_per_company=max_results)
        display_cross_company_analysis_results(results, query)
    else:
        results = system.rag_engine.search_across_companies(
            query=query,
            company_tickers=companies_with_data_filtered,
            top_k_per_company=max_results)
        display_cross_company_raw_results(results, query, "Custom Selection",
                                          include_metadata)


def execute_cross_company_query(system, query, all_companies, use_ai_analysis,
                                max_results, max_companies, include_metadata):
    """Execute query across all companies"""
    st.subheader("🌐 Cross-Company Analysis")

    # Filter to only companies with data
    companies_with_data = get_available_companies()
    companies_with_data_tickers = [
        c['company_ticker'] for c in companies_with_data
    ]

    # Limit companies if needed
    companies_to_search = companies_with_data_tickers[:max_companies] if len(
        companies_with_data_tickers
    ) > max_companies else companies_with_data_tickers

    if len(companies_with_data_tickers) > max_companies:
        st.info(
            f"Analyzing first {max_companies} companies (out of {len(companies_with_data_tickers)} available with data)"
        )

    if use_ai_analysis:
        results = system.analyze_cross_company_query(
            query=query, top_k_per_company=max_results)
        display_cross_company_analysis_results(results, query)
    else:
        results = system.rag_engine.search_across_companies(
            query=query, top_k_per_company=max_results)
        display_cross_company_raw_results(results, query, "All Companies",
                                          include_metadata)


def display_sector_analysis_results(results: Dict, query: str, sector: str):
    """Display sector analysis results"""
    st.markdown('<div class="query-result">', unsafe_allow_html=True)
    st.markdown(f"### 🏭 {sector} Sector M&A Analysis")

    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Sector", sector)

    with col2:
        companies_analyzed = len(results.get('results_by_company', {}))
        st.metric("Companies Analyzed", companies_analyzed)

    with col3:
        total_docs = results.get('total_documents', 0)
        st.metric("Total Documents", total_docs)

    # Analysis content
    if 'analysis' in results:
        st.markdown("### 📊 Sector Analysis")
        st.markdown(results['analysis'])

    st.markdown('</div>', unsafe_allow_html=True)

    # Individual company results
    if 'results_by_company' in results:
        st.subheader("🏢 Company-Specific Results")
        display_company_tabs(results['results_by_company'])


def display_company_tabs(results_by_company: Dict):
    """Display results in company tabs"""
    if not results_by_company:
        st.warning("No results found.")
        return

    # Create tabs for each company
    company_names = list(results_by_company.keys())
    if len(company_names) > 10:
        # If too many companies, show top 10 and a summary
        company_names = company_names[:10]
        st.info(
            f"Showing top 10 companies. Total companies with results: {len(results_by_company)}"
        )

    tabs = st.tabs([f"📈 {name}" for name in company_names])

    for i, (company,
            docs) in enumerate(list(results_by_company.items())[:len(tabs)]):
        with tabs[i]:
            if docs:
                st.write(f"**{len(docs)} documents found**")

                for j, doc in enumerate(docs):
                    with st.expander(
                            f"Document {j+1} (Score: {doc['score']:.3f})"):
                        st.markdown('<div class="document-content">',
                                    unsafe_allow_html=True)
                        content = doc['content']
                        if len(content) > 1000:
                            content = content[:1000] + "..."
                        st.text(content)
                        st.markdown('</div>', unsafe_allow_html=True)

                        if doc.get('metadata'):
                            st.json(doc['metadata'])
            else:
                st.info("No documents found for this company.")


def display_company_analysis_results(results: Dict, query: str):
    """Display AI analysis results for company-specific query"""
    st.subheader("🤖 AI Analysis Results")

    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Company", results.get('company_ticker', 'Unknown'))

    with col2:
        st.metric("Documents Analyzed", results.get('document_count', 0))

    with col3:
        context_size = results.get('context_size_chars', 0)
        st.metric("Context Size", f"{context_size:,} chars")

    # Analysis content
    st.markdown('<div class="query-result">', unsafe_allow_html=True)
    st.markdown("### 📊 Analysis")
    analysis = results.get('analysis', 'No analysis available')
    st.markdown(analysis)
    st.markdown('</div>', unsafe_allow_html=True)

    # Source documents
    if 'source_documents' in results:
        st.subheader("📄 Source Documents")
        for i, doc in enumerate(results['source_documents']):
            with st.expander(f"Document {i+1}"):
                st.markdown('<div class="document-content">',
                            unsafe_allow_html=True)
                content = doc.get('content', doc.get('page_content', ''))
                if len(content) > 1500:
                    content = content[:1500] + "..."
                st.text(content)
                st.markdown('</div>', unsafe_allow_html=True)


def display_raw_documents(documents: List[Dict], query: str, company: str,
                          include_metadata: bool):
    """Display raw document results"""
    st.subheader(f"📄 Raw Documents for {company}")

    if not documents:
        st.warning("No documents found matching the query.")
        return

    st.info(f"Found {len(documents)} documents")

    # Create a table of results
    doc_data = []
    for i, doc in enumerate(documents):
        doc_data.append({
            'Document':
            f"Doc {i+1}",
            'Score':
            f"{doc['score']:.3f}",
            'Content Preview':
            doc['content'][:100] +
            "..." if len(doc['content']) > 100 else doc['content'],
            'Metadata':
            json.dumps(doc.get('metadata', {}), indent=2)
            if doc.get('metadata') and include_metadata else "None"
        })

    df = pd.DataFrame(doc_data)

    # Display table
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_column("Content Preview", wrapText=True, autoHeight=True)
    gb.configure_column("Metadata", hide=True)

    grid_options = gb.build()

    grid_response = AgGrid(df,
                           gridOptions=grid_options,
                           data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                           update_mode=GridUpdateMode.SELECTION_CHANGED,
                           fit_columns_on_grid_load=True,
                           theme='streamlit',
                           height=300)

    # Show selected document details
    if grid_response['selected_rows'] is not None and len(
            grid_response['selected_rows']) > 0:
        selected_idx = int(
            grid_response['selected_rows'][0]['Document'].split()[1]) - 1
        selected_doc = documents[selected_idx]

        st.subheader("📋 Document Details")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="document-content">',
                        unsafe_allow_html=True)
            st.text_area("Full Content:", selected_doc['content'], height=400)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Similarity Score:**")
            st.write(f"{selected_doc['score']:.3f}")

            if selected_doc.get('metadata'):
                st.markdown("**Metadata:**")
                st.json(selected_doc['metadata'])


def display_cross_company_analysis_results(results: Dict, query: str):
    """Display AI analysis results for cross-company query"""
    st.subheader("🤖 Cross-Company AI Analysis")

    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Companies Searched", results.get('companies_searched', 0))

    with col2:
        st.metric("Companies with Results",
                  results.get('companies_with_results', 0))

    with col3:
        st.metric("Total Documents", results.get('total_documents', 0))

    with col4:
        context_size = results.get('context_size_chars', 0)
        st.metric("Context Size", f"{context_size:,} chars")

    # Main analysis
    st.markdown('<div class="query-result">', unsafe_allow_html=True)
    st.markdown("### 📊 Cross-Company Analysis")
    analysis = results.get('analysis', 'No analysis available')
    st.markdown(analysis)
    st.markdown('</div>', unsafe_allow_html=True)

    # Company summaries
    if 'company_summaries' in results and results['company_summaries']:
        st.subheader("🏢 Individual Company Insights")

        for ticker, summary in results['company_summaries'].items():
            with st.expander(f"📈 {ticker}"):
                st.markdown(summary)


def display_cross_company_raw_results(results: Dict, query: str, company: str,
                                      include_metadata: bool):
    """Display raw results for cross-company search"""
    st.subheader(f"📄 Cross-Company Raw Results for {company}")

    if 'error' in results:
        st.error(f"Search failed: {results['error']}")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Companies Searched", results.get('companies_searched', 0))

    with col2:
        st.metric("Companies with Results",
                  results.get('companies_with_results', 0))

    with col3:
        st.metric("Total Documents", results.get('total_documents', 0))

    # Results by company
    results_by_company = results.get('results_by_company', {})

    if not results_by_company:
        st.warning("No documents found matching the query.")
        return

    # Create tabs for each company
    company_tabs = st.tabs(
        [f"📈 {ticker}" for ticker in results_by_company.keys()])

    for i, (ticker, docs) in enumerate(results_by_company.items()):
        with company_tabs[i]:
            if docs:
                st.write(f"**{len(docs)} documents found**")

                for j, doc in enumerate(docs):
                    with st.expander(
                            f"Document {j+1} (Score: {doc['score']:.3f})"):
                        st.markdown('<div class="document-content">',
                                    unsafe_allow_html=True)
                        content = doc['content']
                        if len(content) > 1000:
                            content = content[:1000] + "..."
                        st.text(content)
                        st.markdown('</div>', unsafe_allow_html=True)

                        if doc.get('metadata') and include_metadata:
                            st.json(doc['metadata'])
            else:
                st.info("No documents found for this company.")


def display_skill_shortage_analysis():
    """Display skill shortage analysis page"""
    st.markdown('<h1 class="main-header">👥 Skill Shortage Analysis</h1>', unsafe_allow_html=True)
    
    # Performance monitoring section
    with st.expander("🔧 Performance & System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            system = initialize_system()
            if system:
                st.success("✅ System Ready")
                st.metric("Status", "Initialized")
            else:
                st.error("❌ System Error")
                st.metric("Status", "Failed")
        
        with col2:
            # Check cache status
            companies = get_available_companies()
            st.metric("Companies Cached", len(companies))
            st.metric("Cache TTL", "10 min")
        
        with col3:
            # Show system info
            status = get_system_status()
            if status:
                st.metric("Collections", status.get('total_collections', 0))
                st.metric("Documents", status.get('total_documents', 0))
            else:
                st.metric("Collections", "N/A")
                st.metric("Documents", "N/A")
        
        # Performance tips
        st.info("💡 **Performance Tips:**\n"
                "• First analysis may take 30-60 seconds due to model loading\n"
                "• Subsequent analyses are cached for 30 minutes\n"
                "• System components are cached for 10 minutes\n"
                "• Use the same company/focus area to benefit from caching")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Company Analysis", "Company Rankings", "Sector Comparison", "Trend Analysis", "Pipeline Analysis", "CSV Analysis"]
    )
    
    if analysis_type == "Company Analysis":
        display_company_skill_analysis()
    elif analysis_type == "Company Rankings":
        display_company_skill_rankings()
    elif analysis_type == "Sector Comparison":
        display_sector_skill_comparison()
    elif analysis_type == "Trend Analysis":
        display_skill_trend_analysis()
    elif analysis_type == "Pipeline Analysis":
        display_skill_pipeline_analysis()
    elif analysis_type == "CSV Analysis":
        display_csv_skill_analysis()


# Cache expensive operations with longer TTL
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_skill_analysis(company_ticker, focus_area, include_trends, detailed_analysis):
    """Run skill shortage analysis with caching"""
    system = initialize_system()
    if system:
        try:
            results = system.analyze_company_skill_shortage(
                company_ticker=company_ticker,
                analysis_focus=focus_area if focus_area else None
            )
            return results
        except Exception as e:
            return {'error': str(e)}
    else:
        return {'error': 'System not available'}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_company_rankings_analysis(min_filings, recent_years_only, years_lookback):
    """Run company skill shortage rankings with caching"""
    system = initialize_system()
    if system and hasattr(system, 'skill_shortage_analyzer') and system.skill_shortage_analyzer:
        try:
            # First, try to load cached analysis results
            if not system.skill_shortage_analyzer.results:
                logger.info("No skill shortage analysis results in memory, attempting to load from cache")
                if not system.skill_shortage_analyzer.load_cache():
                    # If no cached results, we need to run the analysis pipeline
                    logger.info("No cached results found, need to run skill shortage analysis pipeline first")
                    return {
                        'error': 'No skill shortage analysis results available. Please run the skill shortage analysis pipeline first.',
                        'suggestion': 'Go to the "Skill Pipeline Analysis" section to run the complete analysis pipeline, or upload analysis results via the "CSV Data Analysis" section.'
                    }
                else:
                    logger.info(f"Loaded {len(system.skill_shortage_analyzer.results)} cached analysis results")
            
            # Check if we have sufficient results after loading
            if not system.skill_shortage_analyzer.results or len(system.skill_shortage_analyzer.results) == 0:
                return {
                    'error': 'No skill shortage analysis results available after loading cache.',
                    'suggestion': 'Please run the skill shortage analysis pipeline first using the "Skill Pipeline Analysis" section.'
                }
            
            logger.info(f"Running company rankings with {len(system.skill_shortage_analyzer.results)} analysis results")
            
            rankings = system.skill_shortage_analyzer.rank_companies_by_skill_shortage(
                min_filings=min_filings,
                recent_years_only=recent_years_only,
                years_lookback=years_lookback
            )
            
            if not rankings:
                return {
                    'error': 'No companies meet the specified criteria for ranking.',
                    'suggestion': 'Try reducing the minimum filings requirement or changing the time period filters.'
                }
            
            return {'rankings': rankings, 'total_companies': len(rankings)}
            
        except Exception as e:
            logger.error(f"Error in company rankings analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    else:
        return {'error': 'Skill shortage analyzer not available. Please check system initialization.'}

def display_company_skill_analysis():
    """Display company-specific skill shortage analysis"""
    st.subheader("🏢 Company Skill Shortage Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Company selection
        companies = get_available_companies()
        if not companies:
            st.warning("No companies available. Please run the setup first.")
            return
            
        company_options = [f"{c['company_ticker']} - {c['company_name']}" for c in companies]
        selected_company = st.selectbox("Select Company", company_options)
        company_ticker = selected_company.split(" - ")[0] if selected_company else None
        
        # Focus area
        focus_area = st.text_input("Focus Area (optional)", 
                                 placeholder="e.g., technology skills, healthcare professionals")
    
    with col2:
        # Analysis options
        st.markdown("**Analysis Options**")
        include_trends = st.checkbox("Include trend analysis", value=True)
        detailed_analysis = st.checkbox("Detailed analysis", value=False)
        
    if st.button("🔍 Analyze Company Skills", type="primary"):
        if company_ticker:
            # Show progress with detailed steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize system (usually cached)
                status_text.text("🔄 Initializing system...")
                progress_bar.progress(10)
                
                # Step 2: Run analysis (cached)
                status_text.text("🔍 Analyzing skill shortages...")
                progress_bar.progress(30)
                
                results = run_skill_analysis(company_ticker, focus_area, include_trends, detailed_analysis)
                
                progress_bar.progress(80)
                status_text.text("📊 Processing results...")
                
                if 'error' not in results:
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed!")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    display_company_skill_results(results, company_ticker)
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Analysis failed: {results['error']}")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please select a company to analyze.")


def display_company_skill_results(results: Dict, company_ticker: str):
    """Display company skill shortage analysis results with AI likelihood scoring"""
    st.success(f"✅ Analysis completed for {company_ticker}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use the correct field name from the analysis results
        documents_analyzed = results.get('chunk_count', results.get('total_chunks_found', 0))
        st.metric("Documents Analyzed", documents_analyzed)
    with col2:
        # Get skill shortage data for likelihood score
        skill_shortage_data = results.get('skill_shortage_data', {})
        likelihood_score = skill_shortage_data.get('average_likelihood_score', 0)
        st.metric("AI Likelihood Score", f"{likelihood_score:.1f}/10")
    with col3:
        # Get keyword mentions from skill_shortage_data if available
        skill_shortage_mentions = skill_shortage_data.get('total_mentions', 0)
        st.metric("Keyword Mentions", skill_shortage_mentions)
    with col4:
        # Risk level based on AI likelihood score
        if likelihood_score >= 7:
            risk_level = "High"
            risk_color = "🔴"
        elif likelihood_score >= 5:
            risk_level = "Medium"
            risk_color = "🟡"
        elif likelihood_score >= 3:
            risk_level = "Low"
            risk_color = "🟢"
        else:
            risk_level = "Very Low"
            risk_color = "🟢"
        st.metric("Risk Level", f"{risk_color} {risk_level}")
    
    # Display AI-generated summary if available
    if skill_shortage_data and skill_shortage_data.get('recent_summary'):
        st.subheader("🤖 AI-Generated Skill Shortage Summary")
        st.markdown(f'<div class="skill-shortage-result">{skill_shortage_data["recent_summary"]}</div>', unsafe_allow_html=True)
    
    # Display skill shortage analysis if available
    if 'skill_shortage_analysis' in results:
        st.subheader("📊 Detailed Skill Shortage Analysis")
        st.markdown(f'<div class="skill-shortage-result">{results["skill_shortage_analysis"]}</div>', unsafe_allow_html=True)
    
    # Display general analysis if no specific skill shortage analysis
    elif 'analysis' in results:
        st.subheader("📊 General Analysis")
        st.markdown(f'<div class="skill-shortage-result">{results["analysis"]}</div>', unsafe_allow_html=True)
    
    # Display document references and source chunks
    if 'document_references' in results:
        st.subheader("📄 Document References & Source Evidence")
        
        doc_refs = results['document_references']
        
        # Document reference summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks Analyzed", doc_refs.get('total_chunks_analyzed', 0))
        with col2:
            doc_summary = doc_refs.get('document_summary', {})
            st.metric("Unique Documents", doc_summary.get('unique_documents', 0))
        with col3:
            if doc_summary.get('highest_similarity_chunk'):
                highest_sim = doc_summary['highest_similarity_chunk']['similarity_score']
                st.metric("Highest Similarity", f"{highest_sim:.3f}")
        
        # Document summary table
        if doc_summary.get('documents'):
            st.subheader("📋 Source Documents Summary")
            
            doc_data = []
            for doc in doc_summary['documents']:
                doc_data.append({
                    'Document ID': doc['document_id'],
                    'Form Type': doc['form_type'],
                    'Filing Date': doc['filing_date'],
                    'Chunks Used': doc['total_chunks'],
                    'Avg Similarity': f"{doc['avg_similarity']:.3f}",
                    'Company': doc['company_name']
                })
            
            doc_df = pd.DataFrame(doc_data)
            st.dataframe(doc_df, use_container_width=True)
        
        # Detailed chunk references
        if doc_refs.get('source_documents'):
            st.subheader("🔍 Detailed Chunk References")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Chunk Summary", "📄 Full Content", "🔗 Metadata"])
            
            with tab1:
                # Chunk summary table
                chunk_data = []
                for chunk in doc_refs['source_documents']:
                    chunk_data.append({
                        'Chunk #': chunk['chunk_index'],
                        'Chunk ID': chunk['chunk_id'],
                        'Similarity': f"{chunk['similarity_score']:.3f}",
                        'Distance': f"{chunk['distance']:.3f}",
                        'Size (chars)': chunk['chunk_size'],
                        'Document ID': chunk['original_document_id'],
                        'Form Type': chunk['form_type'],
                        'Filing Date': chunk['filing_date']
                    })
                
                chunk_df = pd.DataFrame(chunk_data)
                
                # Configure AgGrid for chunk table
                gb = GridOptionsBuilder.from_dataframe(chunk_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_selection('single', use_checkbox=True)
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
                
                grid_options = gb.build()
                
                grid_response = AgGrid(
                    chunk_df,
                    gridOptions=grid_options,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    fit_columns_on_grid_load=True,
                    theme='streamlit',
                    height=400
                )
                
                # Show selected chunk details
                if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
                    selected_chunk_idx = int(grid_response['selected_rows'][0]['Chunk #']) - 1
                    selected_chunk = doc_refs['source_documents'][selected_chunk_idx]
                    
                    st.subheader(f"📋 Chunk {selected_chunk['chunk_index']} Details")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown('<div class="document-content">', unsafe_allow_html=True)
                        st.text_area("Chunk Content:", selected_chunk['content'], height=300, key=f"chunk_content_{selected_chunk_idx}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Chunk Metadata:**")
                        st.write(f"**Similarity Score:** {selected_chunk['similarity_score']:.3f}")
                        st.write(f"**Distance:** {selected_chunk['distance']:.3f}")
                        st.write(f"**Chunk Size:** {selected_chunk['chunk_size']} characters")
                        st.write(f"**Document ID:** {selected_chunk['original_document_id']}")
                        st.write(f"**Form Type:** {selected_chunk['form_type']}")
                        st.write(f"**Filing Date:** {selected_chunk['filing_date']}")
                        
                        if selected_chunk['metadata']:
                            st.markdown("**Full Metadata:**")
                            st.json(selected_chunk['metadata'])
            
            with tab2:
                # Full content view
                st.markdown("**All Chunks Used in Analysis (ordered by similarity):**")
                
                for i, chunk in enumerate(doc_refs['source_documents']):
                    with st.expander(f"Chunk {chunk['chunk_index']} - Similarity: {chunk['similarity_score']:.3f} (ID: {chunk['chunk_id']})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown('<div class="document-content">', unsafe_allow_html=True)
                            st.text(chunk['content'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Reference Info:**")
                            st.write(f"Document: {chunk['original_document_id']}")
                            st.write(f"Form: {chunk['form_type']}")
                            st.write(f"Date: {chunk['filing_date']}")
                            st.write(f"Size: {chunk['chunk_size']} chars")
            
            with tab3:
                # Metadata view
                st.markdown("**Chunk Metadata Details:**")
                
                for i, chunk in enumerate(doc_refs['source_documents']):
                    with st.expander(f"Chunk {chunk['chunk_index']} Metadata"):
                        st.json(chunk['metadata'])
        
        # Date range information
        if doc_summary.get('date_range'):
            date_range = doc_summary['date_range']
            if date_range['earliest'] != 'Unknown' and date_range['latest'] != 'Unknown':
                st.info(f"📅 **Analysis covers filings from {date_range['earliest']} to {date_range['latest']}**")
    
    # Display skill shortage data details if available
    if skill_shortage_data and not skill_shortage_data.get('error'):
        st.subheader("📈 Skill Shortage Details")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filings Analyzed", skill_shortage_data.get('filings_analyzed', 0))
            st.metric("Filings with Keywords", skill_shortage_data.get('filings_with_mentions', 0))
        with col2:
            st.metric("High Likelihood Filings", skill_shortage_data.get('filings_with_high_likelihood', 0))
            avg_keyword_score = skill_shortage_data.get('average_keyword_score', 0)
            st.metric("Avg Keyword Score", f"{avg_keyword_score:.3f}")
        with col3:
            years_data = skill_shortage_data.get('years_with_data', [])
            if years_data:
                st.metric("Years Covered", f"{min(years_data)}-{max(years_data)}")
            else:
                st.metric("Years Covered", "N/A")
        
        # Detailed results if available
        if 'detailed_results' in skill_shortage_data:
            st.subheader("📊 Year-by-Year Analysis")
            
            detailed_data = []
            for result in skill_shortage_data['detailed_results']:
                detailed_data.append({
                    'Year': result.get('year', 'N/A'),
                    'AI Likelihood': f"{result.get('likelihood_score', 0):.1f}/10",
                    'Keyword Mentions': result.get('mentions', 0),
                    'Keyword Score': f"{result.get('keyword_score', 0):.3f}",
                    'Summary Preview': result.get('summary', 'No summary')[:100] + '...' if result.get('summary') else 'No summary'
                })
            
            if detailed_data:
                df = pd.DataFrame(detailed_data)
                st.dataframe(df, use_container_width=True)
    
    # Show if no skill shortage data was found
    elif skill_shortage_data and skill_shortage_data.get('error'):
        st.info(f"ℹ️ {skill_shortage_data['error']}")
        st.info("💡 This analysis is based on document retrieval only. To get skill shortage statistics, run the skill shortage analysis pipeline first.")
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download JSON"):
            st.download_button(
                label="Download Results",
                data=json.dumps(results, indent=2),
                file_name=f"{company_ticker}_skill_analysis.json",
                mime="application/json"
            )
    with col2:
        if st.button("📊 Generate Report"):
            generate_skill_shortage_report(results, company_ticker)


def display_sector_skill_comparison():
    """Display sector skill shortage comparison"""
    st.subheader("🏭 Sector Skill Shortage Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sector selection
        sectors = get_available_sectors()
        if not sectors:
            st.warning("No sector data available.")
            return
            
        selected_sector = st.selectbox("Select Sector", ["All Sectors"] + sectors)
        
        # Company selection for comparison
        companies = get_available_companies()
        if selected_sector != "All Sectors":
            # Filter companies by sector
            sp500_companies = get_sp500_companies()
            sector_companies = [c.symbol for c in sp500_companies if c.sector == selected_sector]
            companies = [c for c in companies if c['company_ticker'] in sector_companies]
        
        company_options = [f"{c['company_ticker']} - {c['company_name']}" for c in companies]
        selected_companies = st.multiselect("Select Companies (optional)", company_options)
        company_tickers = [comp.split(" - ")[0] for comp in selected_companies] if selected_companies else None
    
    with col2:
        st.markdown("**Comparison Options**")
        include_trends = st.checkbox("Include trend data", value=True)
        top_n = st.slider("Top N companies to show", 5, 20, 10)
    
    if st.button("🔍 Compare Skill Shortages", type="primary"):
        with st.spinner("Comparing skill shortages across companies..."):
            system = initialize_system()
            if system:
                try:
                    results = system.compare_skill_shortage_across_companies(
                        company_tickers=company_tickers,
                        sector=selected_sector if selected_sector != "All Sectors" else None
                    )
                    
                    if 'error' not in results:
                        display_sector_comparison_results(results, selected_sector)
                    else:
                        st.error(f"Comparison failed: {results['error']}")
                except Exception as e:
                    st.error(f"Error during comparison: {e}")


def display_sector_comparison_results(results: Dict, sector: str):
    """Display sector comparison results"""
    st.success(f"✅ Comparison completed for {sector}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Companies Compared", results.get('companies_compared', 0))
    with col2:
        avg_score = results.get('avg_severity_score', 0)
        st.metric("Average Severity Score", f"{avg_score:.1f}/10")
    with col3:
        total_mentions = results.get('total_skill_shortage_mentions', 0)
        st.metric("Total Skill Mentions", total_mentions)
    
    # Company rankings
    if 'comparison_data' in results and results['comparison_data']:
        st.subheader("📈 Company Rankings by Skill Shortage Severity")
        
        comparison_df = pd.DataFrame(results['comparison_data'])
        comparison_df = comparison_df.sort_values('severity_score', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            comparison_df.head(15),
            x='company',
            y='severity_score',
            title=f"Skill Shortage Severity Scores - {sector}",
            labels={'severity_score': 'Severity Score (0-10)', 'company': 'Company'},
            color='severity_score',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📊 Detailed Comparison Data")
        st.dataframe(comparison_df, use_container_width=True)


def display_skill_trend_analysis():
    """Display skill shortage trend analysis"""
    st.subheader("📈 Skill Shortage Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Year selection
        current_year = datetime.now().year
        years = st.multiselect(
            "Select Years",
            list(range(2018, current_year + 1)),
            default=[current_year - 2, current_year - 1, current_year]
        )
        
        # Sector selection
        sectors = get_available_sectors()
        selected_sector = st.selectbox("Select Sector (optional)", ["All Sectors"] + sectors)
    
    with col2:
        st.markdown("**Analysis Options**")
        include_predictions = st.checkbox("Include predictions", value=False)
        granularity = st.selectbox("Time Granularity", ["Yearly", "Quarterly"])
    
    if st.button("📊 Analyze Trends", type="primary"):
        if years:
            with st.spinner("Analyzing skill shortage trends..."):
                system = initialize_system()
                if system:
                    try:
                        results = system.analyze_skill_shortage_trends(
                            years=years,
                            sector=selected_sector if selected_sector != "All Sectors" else None
                        )
                        
                        if 'error' not in results:
                            display_trend_analysis_results(results, years, selected_sector)
                        else:
                            st.error(f"Trend analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"Error during trend analysis: {e}")
        else:
            st.warning("Please select at least one year.")


def display_trend_analysis_results(results: Dict, years: List[int], sector: str):
    """Display trend analysis results"""
    st.success(f"✅ Trend analysis completed for {years}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", results.get('data_points', 0))
    with col2:
        st.metric("Companies Covered", results.get('companies_covered', 0))
    with col3:
        st.metric("Years Analyzed", len(results.get('years_covered', [])))
    
    # Trend visualization
    if 'trend_analysis' in results and 'yearly_trends' in results['trend_analysis']:
        st.subheader("📈 Yearly Skill Shortage Trends")
        
        yearly_data = results['trend_analysis']['yearly_trends']
        trend_df = pd.DataFrame([
            {
                'Year': int(year),
                'Total Mentions': data['total_mentions'],
                'Mention Rate': data['mention_rate'] * 100,
                'Companies Affected': data.get('companies_affected', 0)
            }
            for year, data in yearly_data.items()
        ])
        
        # Line chart for trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df['Year'],
            y=trend_df['Mention Rate'],
            mode='lines+markers',
            name='Mention Rate (%)',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title=f"Skill Shortage Mention Rate Over Time - {sector}",
            xaxis_title="Year",
            yaxis_title="Mention Rate (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📊 Trend Data")
        st.dataframe(trend_df, use_container_width=True)


def display_skill_pipeline_analysis():
    """Display comprehensive skill shortage pipeline analysis"""
    st.subheader("🔄 Comprehensive Skill Shortage Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Configuration options
        years = st.multiselect(
            "Select Years",
            list(range(2018, datetime.now().year + 1)),
            default=[datetime.now().year - 1, datetime.now().year]
        )
        
        company_limit = st.slider("Company Limit", 10, 500, 100)
    
    with col2:
        st.markdown("**Pipeline Options**")
        include_trends = st.checkbox("Include trend analysis", value=True)
        include_predictions = st.checkbox("Include predictions", value=False)
        detailed_output = st.checkbox("Detailed output", value=True)
    
    if st.button("🚀 Run Pipeline Analysis", type="primary"):
        if years:
            with st.spinner("Running comprehensive skill shortage analysis pipeline..."):
                system = initialize_system()
                if system:
                    try:
                        results = system.run_skill_shortage_analysis_pipeline(
                            years=years,
                            limit_companies=company_limit
                        )
                        
                        if 'error' not in results:
                            display_pipeline_results(results)
                        else:
                            st.error(f"Pipeline analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"Error during pipeline analysis: {e}")
        else:
            st.warning("Please select at least one year.")


def display_pipeline_results(results: Dict):
    """Display pipeline analysis results"""
    st.success("✅ Pipeline analysis completed successfully!")
    
    # Key metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies Analyzed", results.get('companies_analyzed', 0))
    with col2:
        st.metric("Skill Shortage Mentions", results.get('skill_shortage_mentions', 0))
    with col3:
        st.metric("Critical Skill Gaps", results.get('critical_skill_gaps', 0))
    with col4:
        avg_severity = results.get('avg_severity_score', 0)
        st.metric("Average Severity", f"{avg_severity:.1f}/10")
    
    # Key findings
    if 'key_findings' in results and results['key_findings']:
        st.subheader("🎯 Key Findings")
        for i, finding in enumerate(results['key_findings'][:10], 1):
            st.markdown(f"**{i}.** {finding}")
    
    # Sector breakdown
    if 'sector_breakdown' in results:
        st.subheader("🏭 Sector Breakdown")
        sector_df = pd.DataFrame(results['sector_breakdown'])
        
        fig = px.pie(
            sector_df,
            values='skill_shortage_mentions',
            names='sector',
            title="Skill Shortage Mentions by Sector"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_csv_skill_analysis():
    """Display CSV-based skill shortage analysis"""
    st.subheader("📄 CSV Data Analysis")
    
    st.markdown("""
    Upload a CSV file with company filing data to analyze skill shortages.
    Expected columns: `cik`, `Year`, `FName`, `gvkey`
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        # Preview the data
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Processing Limit", 10, len(df), min(100, len(df)))
        with col2:
            st.metric("Total Rows", len(df))
        
        if st.button("🔍 Analyze CSV Data", type="primary"):
            with st.spinner("Analyzing skill shortages from CSV data..."):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                df.to_csv(temp_path, index=False)
                
                system = initialize_system()
                if system:
                    try:
                        results = system.analyze_from_csv_data(
                            csv_path=temp_path,
                            limit=limit
                        )
                        
                        if 'error' not in results:
                            display_csv_analysis_results(results)
                        else:
                            st.error(f"CSV analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"Error during CSV analysis: {e}")
                    finally:
                        # Clean up temp file
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)


def display_csv_analysis_results(results: Dict):
    """Display CSV analysis results"""
    st.success("✅ CSV analysis completed!")
    
    # Summary statistics
    if 'summary_stats' in results:
        stats = results['summary_stats']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Filings Analyzed", stats.get('total_filings_analyzed', 0))
        with col2:
            st.metric("Filings with Skill Mentions", stats.get('filings_with_skill_shortage_mentions', 0))
        with col3:
            mention_rate = stats.get('skill_shortage_mention_rate', 0)
            st.metric("Mention Rate", f"{mention_rate:.1%}")
    
    # Analysis results
    if 'analysis_results' in results:
        st.subheader("📊 Analysis Results")
        st.write(results['analysis_results'])
    
    # Significant findings
    if 'significant_findings' in results:
        st.subheader("🎯 Significant Findings")
        st.write(results['significant_findings'])


def display_trends_analytics():
    """Display trends and analytics dashboard"""
    st.markdown('<h1 class="main-header">📈 Trends & Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["M&A Trends", "Skill Shortage Trends", "Comparative Analytics"])
    
    with tab1:
        display_ma_trends_tab()
    
    with tab2:
        display_skill_trends_tab()
    
    with tab3:
        display_comparative_analytics_tab()


def display_ma_trends_tab():
    """Display M&A trends analysis"""
    st.subheader("🔄 M&A Organizational Impact Trends")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        years = st.multiselect(
            "Select Years",
            list(range(2018, datetime.now().year + 1)),
            default=[datetime.now().year - 2, datetime.now().year - 1]
        )
        
        sectors = get_available_sectors()
        selected_sector = st.selectbox("Select Sector", ["All Sectors"] + sectors)
    
    with col2:
        st.markdown("**Analysis Options**")
        include_predictions = st.checkbox("Include predictions", value=False)
    
    if st.button("📊 Analyze M&A Trends", type="primary"):
        if years:
            with st.spinner("Analyzing M&A trends..."):
                system = initialize_system()
                if system:
                    try:
                        results = system.analyze_trends(
                            years=years,
                            sector=selected_sector if selected_sector != "All Sectors" else None
                        )
                        
                        if 'error' not in results:
                            display_ma_trend_results(results, years, selected_sector)
                        else:
                            st.error(f"M&A trend analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"Error during M&A trend analysis: {e}")


def display_ma_trend_results(results: Dict, years: List[int], sector: str):
    """Display M&A trend analysis results"""
    st.success(f"✅ M&A trend analysis completed for {sector}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", results.get('data_points', 0))
    with col2:
        st.metric("Companies Covered", results.get('companies_covered', 0))
    with col3:
        st.metric("Years Analyzed", len(results.get('years_covered', [])))
    
    # Trend visualization
    if 'trend_analysis' in results and 'yearly_trends' in results['trend_analysis']:
        yearly_data = results['trend_analysis']['yearly_trends']
        trend_df = pd.DataFrame([
            {
                'Year': int(year),
                'Total Events': data['total_events'],
                'Avg Impact Score': data['avg_impact_score']
            }
            for year, data in yearly_data.items()
        ])
        
        fig = px.line(
            trend_df,
            x='Year',
            y='Avg Impact Score',
            title=f"M&A Impact Trends - {sector}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)


def display_skill_trends_tab():
    """Display skill shortage trends tab"""
    st.subheader("👥 Skill Shortage Trends")
    st.info("Use the Skill Shortage Analysis page for detailed trend analysis.")


def display_comparative_analytics_tab():
    """Display comparative analytics"""
    st.subheader("⚖️ Comparative Analytics")
    
    st.markdown("""
    Compare M&A organizational impacts with skill shortage patterns to identify correlations
    and insights across companies and sectors.
    """)
    
    if st.button("🔍 Run Comparative Analysis", type="primary"):
        with st.spinner("Running comparative analysis..."):
            # This would implement cross-analysis between M&A and skill shortage data
            st.info("Comparative analysis feature coming soon!")


def display_system_dashboard():
    """Display comprehensive system dashboard"""
    st.markdown('<h1 class="main-header">⚙️ System Dashboard</h1>', unsafe_allow_html=True)
    
    # System status
    status = get_system_status()
    
    if status:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vector_status = status.get('vector_store', {})
            st.metric("Company Collections", vector_status.get('company_collections', 0))
        
        with col2:
            st.metric("Total Documents", vector_status.get('total_documents', 0))
        
        with col3:
            sp500_status = status.get('sp500_data', {})
            st.metric("S&P 500 Companies", sp500_status.get('companies_loaded', 0))
        
        with col4:
            edgar_status = status.get('edgar_data', {})
            st.metric("EDGAR Filings", edgar_status.get('filings_cached', 0))
        
        # Detailed status information
        st.subheader("📊 Detailed System Information")
        
        tab1, tab2, tab3 = st.tabs(["Vector Store", "Data Sources", "System Health"])
        
        with tab1:
            st.json(vector_status)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**S&P 500 Data**")
                st.json(sp500_status)
            with col2:
                st.markdown("**EDGAR Data**")
                st.json(edgar_status)
        
        with tab3:
            st.success("✅ All systems operational")
            st.info("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        st.error("❌ System status unavailable")


def generate_skill_shortage_report(results: Dict, company_ticker: str):
    """Generate a comprehensive skill shortage report with AI likelihood scoring"""
    # Get the correct document count
    documents_analyzed = results.get('chunk_count', results.get('total_chunks_found', 0))
    
    # Get skill shortage data
    skill_shortage_data = results.get('skill_shortage_data', {})
    skill_shortage_mentions = skill_shortage_data.get('total_mentions', 0)
    likelihood_score = skill_shortage_data.get('average_likelihood_score', 0)
    recent_summary = skill_shortage_data.get('recent_summary', '')
    
    # Determine risk level
    if likelihood_score >= 7:
        risk_level = "High Risk"
    elif likelihood_score >= 5:
        risk_level = "Medium Risk"
    elif likelihood_score >= 3:
        risk_level = "Low Risk"
    else:
        risk_level = "Very Low Risk"
    
    report = f"""
# Skill Shortage Analysis Report
## Company: {company_ticker}
## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Executive Summary
- Documents Analyzed: {documents_analyzed}
- **AI Likelihood Score: {likelihood_score:.1f}/10**
- **Risk Level: {risk_level}**
- Keyword Mentions: {skill_shortage_mentions}

### AI-Generated Summary
{recent_summary if recent_summary else 'No AI summary available'}

### Analysis Details
"""
    
    # Add skill shortage analysis if available
    if 'skill_shortage_analysis' in results:
        report += f"\n#### Detailed Skill Shortage Analysis:\n{results['skill_shortage_analysis']}\n"
    elif 'analysis' in results:
        report += f"\n#### General Analysis:\n{results['analysis']}\n"
    
    # Add document references section
    if 'document_references' in results:
        doc_refs = results['document_references']
        report += f"\n### Document References & Source Evidence\n"
        report += f"- Total chunks analyzed: {doc_refs.get('total_chunks_analyzed', 0)}\n"
        
        doc_summary = doc_refs.get('document_summary', {})
        if doc_summary and not doc_summary.get('error'):
            report += f"- Unique source documents: {doc_summary.get('unique_documents', 0)}\n"
            
            # Add date range
            date_range = doc_summary.get('date_range', {})
            if date_range.get('earliest') != 'Unknown' and date_range.get('latest') != 'Unknown':
                report += f"- Filing date range: {date_range['earliest']} to {date_range['latest']}\n"
            
            # Add source documents
            documents = doc_summary.get('documents', [])
            if documents:
                report += f"\n#### Source Documents:\n"
                for i, doc in enumerate(documents, 1):
                    report += f"{i}. **{doc['document_id']}**\n"
                    report += f"   - Form Type: {doc['form_type']}\n"
                    report += f"   - Filing Date: {doc['filing_date']}\n"
                    report += f"   - Chunks Used: {doc['total_chunks']}\n"
                    report += f"   - Average Similarity: {doc['avg_similarity']:.3f}\n\n"
            
            # Add highest similarity chunk
            highest_chunk = doc_summary.get('highest_similarity_chunk')
            if highest_chunk:
                report += f"\n#### Highest Similarity Evidence:\n"
                report += f"- Similarity Score: {highest_chunk['similarity_score']:.3f}\n"
                report += f"- Chunk ID: {highest_chunk['id']}\n"
        
        # Add detailed chunk references
        source_docs = doc_refs.get('source_documents', [])
        if source_docs:
            report += f"\n#### Detailed Source Evidence:\n"
            
            # Sort by similarity and show top chunks
            sorted_chunks = sorted(source_docs, key=lambda x: x['similarity_score'], reverse=True)
            for i, chunk in enumerate(sorted_chunks[:5], 1):  # Top 5 chunks
                report += f"\n**Evidence Chunk {i}:**\n"
                report += f"- Chunk ID: {chunk['chunk_id']}\n"
                report += f"- Similarity Score: {chunk['similarity_score']:.3f}\n"
                report += f"- Document: {chunk['original_document_id']}\n"
                report += f"- Form Type: {chunk['form_type']}\n"
                report += f"- Filing Date: {chunk['filing_date']}\n"
                report += f"- Content Preview: {chunk['content'][:200]}{'...' if len(chunk['content']) > 200 else ''}\n"
    
    # Add skill shortage data details
    if skill_shortage_data and not skill_shortage_data.get('error'):
        report += f"""
### Skill Shortage Statistics:
- Filings Analyzed: {skill_shortage_data.get('filings_analyzed', 0)}
- Filings with Keywords: {skill_shortage_data.get('filings_with_mentions', 0)}
- High Likelihood Filings: {skill_shortage_data.get('filings_with_high_likelihood', 0)}
- Average Keyword Score: {skill_shortage_data.get('average_score', 0):.4f}
- Average AI Likelihood Score: {likelihood_score:.1f}/10
"""
        years_data = skill_shortage_data.get('years_with_data', [])
        if years_data:
            report += f"- Years Covered: {min(years_data)}-{max(years_data)}\n"
        
        # Add detailed results if available
        detailed_results = skill_shortage_data.get('detailed_results', [])
        if detailed_results:
            report += f"\n#### Year-by-Year Analysis:\n"
            for result in detailed_results:
                year = result.get('year', 'Unknown')
                ai_score = result.get('likelihood_score', 0)
                keyword_mentions = result.get('keyword_mentions', 0)
                keyword_score = result.get('keyword_score', 0)
                summary = result.get('summary', '')
                
                report += f"\n**{year}:**\n"
                report += f"- AI Likelihood Score: {ai_score:.1f}/10\n"
                report += f"- Keyword Mentions: {keyword_mentions}\n"
                report += f"- Keyword Score: {keyword_score:.4f}\n"
                if summary:
                    report += f"- Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}\n"
    
    elif skill_shortage_data and skill_shortage_data.get('error'):
        report += f"\n#### Note:\n{skill_shortage_data['error']}\n"
        report += "This analysis is based on document retrieval only. To get detailed skill shortage statistics, run the skill shortage analysis pipeline first.\n"
    
    # Add methodology section
    report += f"""
### Methodology
This analysis was conducted using the Financial Graph RAG system, which:
1. Retrieved relevant document chunks from SEC filings
2. Ranked chunks by semantic similarity to skill shortage queries
3. Used AI analysis to interpret the evidence
4. Provided source references for transparency and verification

### Source Verification
All findings are backed by specific document chunks from SEC filings. 
Chunk IDs and similarity scores are provided for verification and further investigation.
"""
    
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"{company_ticker}_skill_shortage_report.md",
        mime="text/markdown"
    )


def display_company_skill_rankings():
    """Display company skill shortage rankings interface"""
    st.subheader("🏆 Company Skill Shortage Rankings")
    
    st.markdown("""
    Rank all companies by their skill shortage levels from highest to lowest based on similarity analysis 
    of SEC filings using AI likelihood scores and keyword patterns.
    """)
    
    # Check analysis data availability and show status
    analysis_status = check_skill_shortage_analysis_availability()
    
    if analysis_status['available']:
        st.success(f"✅ Analysis data available: {analysis_status['count']} company filings analyzed (loaded from {analysis_status['source']})")
    else:
        st.warning("⚠️ No skill shortage analysis data available")
        if analysis_status.get('error'):
            st.error(f"Error: {analysis_status['error']}")
        
        st.info("""
        💡 **To generate company rankings, you need to run the skill shortage analysis first:**
        
        1. Go to the **"Pipeline Analysis"** tab above
        2. Click "🚀 Run Complete Analysis Pipeline" 
        3. Wait for analysis to complete
        4. Return here to generate rankings
        
        Or upload existing analysis data via the **"CSV Data Analysis"** tab.
        """)
    
    # Configuration options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time period options
        recent_years_only = st.checkbox(
            "Recent years only", 
            value=True,
            help="Focus on recent filings for more current skill shortage indicators"
        )
        
        if recent_years_only:
            years_lookback = st.slider("Years to look back", 1, 5, 3)
        else:
            years_lookback = 10  # Default fallback
    
    with col2:
        st.markdown("**Filtering Options**")
        min_filings = st.slider(
            "Min filings per company", 
            1, 10, 1,
            help="Minimum number of filings required for a company to be included"
        )
        
        # Display options
        show_summaries = st.checkbox("Show AI summaries", value=True)
        max_display = st.slider("Max companies to display", 10, 100, 25)
    
    # Information about similarity-based ranking
    with st.expander("ℹ️ About Similarity-Based Ranking"):
        st.markdown("""
        **How Similarity Ranking Works:**
        
        The ranking uses a similarity-based scoring system that combines:
        - **AI Likelihood Scores (60%)**: Semantic analysis of filing language for skill shortage patterns
        - **Peak Similarity (20%)**: Highest likelihood score across all filings for the company
        - **Mention Consistency (30%)**: Rate of skill shortage mentions across filings
        - **High-Confidence Rate (20%)**: Proportion of filings with high likelihood scores (>6.0)
        
        **Similarity Score Interpretation:**
        - **8.0+**: Very high similarity to skill shortage patterns
        - **6.0-7.9**: High similarity 
        - **4.0-5.9**: Moderate similarity
        - **2.0-3.9**: Low similarity
        - **<2.0**: Very low similarity
        
        This approach provides a comprehensive measure of how closely each company's SEC filings match known skill shortage patterns.
        """)
    
    if st.button("🚀 Generate Company Rankings", type="primary"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing skill shortage analyzer...")
            progress_bar.progress(10)
            
            status_text.text("📊 Analyzing company skill shortage similarity...")
            progress_bar.progress(30)
            
            # Run the ranking analysis
            results = run_company_rankings_analysis(
                min_filings=min_filings,
                recent_years_only=recent_years_only,
                years_lookback=years_lookback
            )
            
            progress_bar.progress(80)
            status_text.text("📈 Processing rankings...")
            
            if 'error' not in results:
                progress_bar.progress(100)
                status_text.text("✅ Rankings generated successfully!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                display_company_rankings_results(
                    results['rankings'], 
                    show_summaries, 
                    max_display,
                    results['total_companies']
                )
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ {results['error']}")
                
                # Show helpful information and suggestions
                if 'suggestion' in results:
                    st.info(f"💡 **Suggestion:** {results['suggestion']}")
                
                st.markdown("""
                ### 📋 How to Get Company Rankings Working:
                
                **Option 1: Run the Complete Analysis Pipeline**
                1. Go to the **"Pipeline Analysis"** tab above
                2. Click "🚀 Run Complete Analysis Pipeline"
                3. Wait for the analysis to complete (this may take several minutes)
                4. Return here to generate company rankings
                
                **Option 2: Upload Existing Analysis Data**
                1. Go to the **"CSV Data Analysis"** tab above
                2. Upload a CSV file with columns: `cik`, `Year`, `FName`, `gvkey`
                3. Run the analysis on your data
                4. Return here to generate company rankings
                
                **What the Analysis Does:**
                - Downloads and analyzes SEC 10-K filings
                - Uses AI to detect skill shortage patterns in corporate language
                - Generates likelihood scores (0-10) for skill shortage probability
                - Counts explicit skill shortage keyword mentions
                - Creates a comprehensive database for company ranking
                
                **Note:** The initial analysis can take 10-30 minutes depending on the number of companies and filings processed.
                """)
                
                # Show current system status
                with st.expander("🔍 System Status Details"):
                    system_status = get_system_status()
                    if system_status.get('status') == 'ready':
                        st.success("✅ System is ready and initialized")
                        st.info("The system is working properly, but no skill shortage analysis data is available yet.")
                    else:
                        st.warning("⚠️ System may not be fully initialized")
                        st.json(system_status)
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error generating rankings: {e}")


def display_company_rankings_results(rankings, show_summaries, max_display, total_companies):
    """Display the company skill shortage rankings results"""
    st.success(f"✅ Successfully ranked {total_companies} companies by skill shortage similarity")
    
    if not rankings:
        st.warning("No companies found meeting the specified criteria.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies Ranked", len(rankings))
    
    with col2:
        # Calculate companies with high similarity scores
        high_score_companies = len([r for r in rankings if r['similarity_score'] >= 6.0])
        st.metric("High Similarity (≥6.0)", high_score_companies)
    
    with col3:
        avg_filings = sum(r['filings_analyzed'] for r in rankings) / len(rankings)
        st.metric("Avg Filings per Company", f"{avg_filings:.1f}")
    
    with col4:
        # Show average similarity score
        avg_similarity = sum(r['similarity_score'] for r in rankings) / len(rankings)
        st.metric("Avg Similarity Score", f"{avg_similarity:.2f}")
    
    # Top 10 companies visualization
    st.subheader("📊 Top 10 Companies with Highest Skill Shortage Similarity")
    
    top_10 = rankings[:10]
    
    # Create similarity-based visualization
    y_values = [r['similarity_score'] for r in top_10]
    y_title = "Similarity Score"
    
    fig = px.bar(
        x=[r['ticker'] for r in top_10],
        y=y_values,
        title="Top 10 Companies - Skill Shortage Similarity Score",
        labels={'x': 'Company Ticker', 'y': y_title},
        color=y_values,
        color_continuous_scale='Viridis',
        text=[f"{val:.2f}" for val in y_values]
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed rankings table
    st.subheader(f"📋 Detailed Company Rankings (Top {min(max_display, len(rankings))})")
    
    # Prepare data for display
    display_rankings = rankings[:max_display]
    
    # Create comprehensive data table
    table_data = []
    for company in display_rankings:
        row = {
            'Rank': company['rank'],
            'Ticker': company['ticker'],
            'Company Name': company['company_name'][:40] + '...' if len(company['company_name']) > 40 else company['company_name'],
            'Similarity Score': f"{company['similarity_score']:.2f}",
            'Filings': company['filings_analyzed'],
            'Years': f"{min(company['years_analyzed'])}-{max(company['years_analyzed'])}" if company['years_analyzed'] else "N/A",
            'AI Likelihood': f"{company['avg_likelihood_score']:.1f}/10",
            'Max Likelihood': f"{company['max_likelihood_score']:.1f}/10",
            'Keyword Mentions': company['total_mentions'],
            'Avg Mentions/Filing': f"{company['avg_mentions_per_filing']:.2f}",
            'Mention Rate': f"{company['mention_rate']:.3f}",
            'High Likelihood Rate': f"{company['high_likelihood_rate']:.3f}"
        }
        table_data.append(row)
    
    # Display the table with AgGrid
    df = pd.DataFrame(table_data)
    
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    
    # Highlight similarity score column
    gb.configure_column("Similarity Score", cellStyle={'backgroundColor': '#e8f5e8'})
    
    grid_options = gb.build()
    
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        theme='streamlit',
        height=500
    )
    
    # Show detailed information for selected company
    if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
        selected_row = grid_response['selected_rows'][0]
        selected_rank = int(selected_row['Rank'])
        selected_company = rankings[selected_rank - 1]  # Ranks are 1-indexed
        
        display_selected_company_ranking_details(selected_company, show_summaries)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Rankings CSV"):
            # Create export data
            export_data = []
            for company in rankings:
                export_data.append({
                    'rank': company['rank'],
                    'ticker': company['ticker'],
                    'company_name': company['company_name'],
                    'cik': company['cik'],
                    'filings_analyzed': company['filings_analyzed'],
                    'years_analyzed': ', '.join(map(str, company['years_analyzed'])),
                    'similarity_score': company['similarity_score'],
                    'avg_likelihood_score': company['avg_likelihood_score'],
                    'max_likelihood_score': company['max_likelihood_score'],
                    'total_mentions': company['total_mentions'],
                    'avg_mentions_per_filing': company['avg_mentions_per_filing'],
                    'mention_rate': company['mention_rate'],
                    'high_likelihood_rate': company['high_likelihood_rate'],
                    'recent_summary': company['recent_summary'] if show_summaries else ''
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"skill_shortage_rankings_similarity_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Full Report"):
            # Generate comprehensive report
            report = generate_rankings_report(rankings, total_companies)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"skill_shortage_rankings_report_similarity_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col3:
        if st.button("🔄 Refresh Rankings"):
            st.cache_data.clear()
            st.experimental_rerun()


def display_selected_company_ranking_details(company, show_summaries):
    """Display detailed information for selected company from rankings"""
    st.subheader(f"🏢 Detailed Analysis: {company['ticker']} - {company['company_name']}")
    
    # Company metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rank", f"#{company['rank']}")
        st.metric("Filings Analyzed", company['filings_analyzed'])
    
    with col2:
        st.metric("Similarity Score", f"{company['similarity_score']:.2f}")
        st.metric("AI Likelihood Score", f"{company['avg_likelihood_score']:.1f}/10")
    
    with col3:
        st.metric("Max Likelihood", f"{company['max_likelihood_score']:.1f}/10")
        st.metric("Total Mentions", company['total_mentions'])
    
    with col4:
        st.metric("Avg Mentions/Filing", f"{company['avg_mentions_per_filing']:.2f}")
        st.metric("Mention Rate", f"{company['mention_rate']:.3f}")
    
    # Risk assessment
    likelihood_score = company['avg_likelihood_score']
    if likelihood_score >= 7:
        risk_level = "🔴 High Risk"
        risk_color = "error"
    elif likelihood_score >= 5:
        risk_level = "🟡 Medium Risk"
        risk_color = "warning"
    elif likelihood_score >= 3:
        risk_level = "🟢 Low Risk"
        risk_color = "success"
    else:
        risk_level = "🟢 Very Low Risk"
        risk_color = "success"
    
    if risk_color == "error":
        st.error(f"**Risk Assessment:** {risk_level}")
    elif risk_color == "warning":
        st.warning(f"**Risk Assessment:** {risk_level}")
    else:
        st.success(f"**Risk Assessment:** {risk_level}")
    
    # Years analyzed
    if company['years_analyzed']:
        years_str = f"{min(company['years_analyzed'])}-{max(company['years_analyzed'])}"
        st.info(f"📅 **Analysis Period:** {years_str} ({len(company['years_analyzed'])} years)")
    
    # Filing statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Filing Statistics:**")
        mention_rate = company['mention_rate']
        high_likelihood_rate = company['high_likelihood_rate']
        
        st.write(f"• Filings with mentions: {company['filings_with_mentions']} ({mention_rate:.1%})")
        st.write(f"• High likelihood filings: {company['filings_with_high_likelihood']} ({high_likelihood_rate:.1%})")
    
    with col2:
        st.markdown("**Scoring Details:**")
        st.write(f"• Similarity score: {company['similarity_score']:.2f}")
        st.write(f"• Average likelihood: {company['avg_likelihood_score']:.1f}/10")
        st.write(f"• Maximum likelihood: {company['max_likelihood_score']:.1f}/10")
        st.write(f"• High likelihood rate: {company['high_likelihood_rate']:.3f}")
    
    # AI Summary
    if show_summaries and company['recent_summary']:
        st.subheader("🤖 AI-Generated Summary")
        st.markdown(f'<div class="skill-shortage-result">{company["recent_summary"]}</div>', unsafe_allow_html=True)
    elif show_summaries:
        st.info("No AI-generated summary available for this company.")


def generate_rankings_report(rankings, total_companies):
    """Generate a comprehensive rankings report"""
    
    report = f"""# Company Skill Shortage Rankings Report

## Analysis Summary
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Ranking Method:** Similarity-Based Analysis
- **Total Companies Ranked:** {total_companies}
- **Analysis Period:** Based on available SEC filings

## Methodology
This ranking analysis uses the Financial Graph RAG system to analyze SEC filings for skill shortage indicators:

1. **AI Likelihood Scoring (0-10):** Advanced language model analysis of filing content
2. **Keyword Detection:** Traditional pattern matching for skill shortage terms
3. **Similarity Analysis:** Combines AI likelihood scores with keyword patterns to create a comprehensive similarity score
4. **Peak Similarity:** Identifies companies with the highest skill shortage indicators
5. **Mention Consistency:** Evaluates consistency of skill shortage mentions across filings

## Top 20 Companies with Highest Skill Shortage Indicators

| Rank | Ticker | Company | Similarity Score | AI Score | Mentions |
|------|--------|---------|------------------|----------|----------|"""
    
    # Add top 20 companies to the report
    for i, company in enumerate(rankings[:20]):
        report += f"\n| {company['rank']} | {company['ticker']} | {company['company_name'][:30]}{'...' if len(company['company_name']) > 30 else ''} | {company['similarity_score']:.2f} | {company['avg_likelihood_score']:.1f} | {company['total_mentions']} |"
    
    # Add key findings
    if rankings:
        high_similarity_companies = [c for c in rankings if c['similarity_score'] >= 6.0]
        avg_similarity = sum(c['similarity_score'] for c in rankings) / len(rankings)
        
        report += f"""

## Key Findings

### Overall Statistics
- **Average Similarity Score:** {avg_similarity:.2f}
- **Companies with High Similarity (≥6.0):** {len(high_similarity_companies)} ({len(high_similarity_companies)/len(rankings)*100:.1f}%)
- **Highest Similarity Score:** {rankings[0]['similarity_score']:.2f} ({rankings[0]['ticker']})
- **Most Filings Analyzed:** {max(c['filings_analyzed'] for c in rankings)} filings

### Top Performers
"""
        
        # Add top 5 companies with details
        for i, company in enumerate(rankings[:5]):
            report += f"""
**{i+1}. {company['ticker']} - {company['company_name']}**
- Similarity Score: {company['similarity_score']:.2f}
- AI Likelihood Score: {company['avg_likelihood_score']:.1f}/10
- Total Mentions: {company['total_mentions']}
- Filings Analyzed: {company['filings_analyzed']}
- Analysis Years: {min(company['years_analyzed']) if company['years_analyzed'] else 'N/A'}-{max(company['years_analyzed']) if company['years_analyzed'] else 'N/A'}
"""
    
    report += f"""

## Analysis Methodology Details

### Similarity Scoring
The similarity score combines multiple factors to identify companies with the highest skill shortage indicators:

- **AI Likelihood Component:** Uses advanced language models to assess the likelihood of skill shortages
- **Keyword Pattern Component:** Identifies specific terms and phrases related to skill shortages
- **Peak Similarity Analysis:** Identifies companies with the most pronounced skill shortage signals
- **Consistency Metrics:** Evaluates how consistently skill shortage indicators appear across filings

### Data Sources
- **SEC EDGAR Filings:** 10-K, 10-Q, and other regulatory filings
- **Analysis Period:** Multiple years of filing data for comprehensive trend analysis
- **AI Processing:** Advanced natural language processing for context-aware analysis

### Limitations and Disclaimers
- Analysis is based on publicly available SEC filings only
- AI scoring may not capture all nuances of business context
- Rankings should be considered alongside other business intelligence
- Past filing content may not reflect current company status
- This analysis is for informational purposes only and should not be used as the sole basis for investment decisions

---
*Report generated by Financial Graph RAG System - Skill Shortage Analyzer*
*Generation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return report


# Hiring Difficulties Analysis Functions

@st.cache_data(ttl=60) # Cache for 1 minute
def check_hiring_difficulties_analysis_availability():
    """Check if hiring difficulties analysis data is available by checking for the output file."""
    try:
        output_file = Path(settings.data_directory) / "hiring_difficulties_summary.csv"
        if output_file.exists():
            df = pd.read_csv(output_file)
            return {
                'available': True,
                'total_companies': df['cik'].nunique(),
                'total_filings': len(df),
                'source': 'CSV file',
                'last_updated': datetime.fromtimestamp(output_file.stat().st_mtime)
            }
        else:
            return {
                'available': False,
                'reason': 'No analysis results file found',
                'suggestion': 'Run the analysis pipeline in the "Run Full Pipeline" tab.'
            }
    except Exception as e:
        logger.error(f"Error checking hiring difficulties analysis availability: {e}")
        return {
            'available': False,
            'reason': f'Error reading cache file: {str(e)}',
            'suggestion': 'Check system logs for details.'
        }


def display_hiring_difficulties_analysis():
    """Display hiring difficulties analysis page"""
    st.markdown('<h1 class="main-header">🔍 Hiring Difficulties Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Analyze company SEC filings for hiring difficulties using AI-powered summarization.")

    tab1, tab2 = st.tabs(["Company Analysis", "Run Full Pipeline"])

    with tab1:
        display_company_hiring_analysis()

    with tab2:
        display_hiring_pipeline_analysis()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_hiring_analysis(company_ticker: str, years: List[int]):
    """Run hiring difficulties analysis for a single company."""
    system = initialize_system()
    if system:
        try:
            # This method is assumed to be in FinancialVectorRAG, calling the new analyzer
            # It should fetch filings for the company and pass them to the analyzer.
            results = system.analyze_filings_for_hiring_difficulties(
                company_ticker=company_ticker,
                years=years
            )
            return results
        except AttributeError:
             return {'error': 'The method to analyze single-company hiring difficulties is not implemented in the backend.'}
        except Exception as e:
            return {'error': str(e)}
    else:
        return {'error': 'System not available'}


def display_company_hiring_analysis():
    """Display company-specific hiring difficulties analysis"""
    st.subheader("🏢 Single Company Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Company selection
        companies = get_available_companies()
        if not companies:
            st.warning("No companies available. Please run the data setup first.")
            return
            
        company_options = [f"{c['company_ticker']} - {c['company_name']}" for c in companies]
        selected_company = st.selectbox("Select Company", company_options, key="hiring_company_select")
        company_ticker = selected_company.split(" - ")[0] if selected_company else None
        
    with col2:
        # Analysis options
        st.markdown("**Analysis Options**")
        years = st.multiselect(
            "Select Years to Analyze",
            list(range(2020, datetime.now().year + 1)),
            default=[datetime.now().year - 1, datetime.now().year],
            key="hiring_years_select"
        )
        
    if st.button("🔍 Analyze Company", type="primary"):
        if company_ticker and years:
            with st.spinner(f"Analyzing {company_ticker} for years {years}..."):
                # I'm assuming analyze_filings_for_hiring_difficulties exists on the system object.
                # It should return a list of dicts.
                results = run_hiring_analysis(company_ticker, years)
                
                if isinstance(results, dict) and 'error' in results:
                    st.error(f"Analysis failed: {results['error']}")
                else:
                    display_company_hiring_results(results, company_ticker)
        else:
            st.warning("Please select a company and at least one year to analyze.")


def display_company_hiring_results(results: List[Dict], company_ticker: str):
    """Display company hiring difficulties analysis results."""
    st.success(f"✅ Analysis completed for {company_ticker}")

    if not results:
        st.warning("No analysis results found for the selected criteria.")
        return

    # Overall metrics
    mentioned_count = sum(1 for r in results if r.get("is_mentioned"))
    error_count = sum(1 for r in results if r.get("error"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filings Analyzed", len(results))
    with col2:
        st.metric("Filings with Mentions", mentioned_count)
    with col3:
        st.metric("Analysis Errors", error_count)

    # Display each summary, sorted by year
    sorted_results = sorted(results, key=lambda r: r.get('year', 0), reverse=True)

    for result in sorted_results:
        expander_title = f"📄 {result.get('year', 'N/A')} Filing for {result.get('ticker', 'N/A')}"
        with st.expander(expander_title, expanded=result.get("is_mentioned", False)):
            if result.get("error"):
                st.error(f"Analysis failed: {result['error']}")
            elif result.get("is_mentioned"):
                st.markdown('<div class="skill-shortage-result" style="background-color: #fff3cd; border-left-color: #ffc107;">', unsafe_allow_html=True)
                st.markdown(result.get("summary", "No summary available."))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No significant hiring difficulties mentioned.")
                if result.get("summary"):
                    st.markdown(f'<div class="document-content">{result.get("summary")}</div>', unsafe_allow_html=True)
            
            st.caption(f"CIK: {result.get('cik')} | Filing URL: [Link]({result.get('filing_url')})")


def display_hiring_pipeline_analysis():
    """Display hiring difficulties pipeline analysis"""
    st.subheader("🚀 Run Full Analysis Pipeline")
    st.markdown("""
    Run the complete analysis pipeline to process SEC filings for multiple companies and years.
    This will generate a summary CSV file with the results, which can be used for further analysis.
    This process can take a significant amount of time depending on the number of filings.
    """)

    # Check for existing data
    analysis_status = check_hiring_difficulties_analysis_availability()
    if analysis_status.get('available'):
        st.success("✅ Analysis data file found.")
        with st.expander("Data File Details"):
            st.metric("Total Companies", analysis_status.get('total_companies', 'N/A'))
            st.metric("Total Filings", analysis_status.get('total_filings', 'N/A'))
            if analysis_status.get('last_updated'):
                st.metric("Last Updated", analysis_status['last_updated'].strftime("%Y-%m-%d %H:%M:%S"))
    else:
        st.warning("⚠️ No analysis data file found. Run the pipeline to generate it.")

    with st.form("pipeline_run_form"):
        st.markdown("**Pipeline Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            years = st.multiselect(
                "Select Years to Analyze",
                options=list(range(2020, datetime.now().year + 1)),
                default=[2022, 2023]
            )
        with col2:
            limit_companies = st.number_input(
                "Limit Companies (optional)",
                min_value=1,
                max_value=500,
                value=None,
                placeholder="All S&P 500"
            )

        submitted = st.form_submit_button("🚀 Run Analysis Pipeline", type="primary")

        if submitted:
            if not years:
                st.error("Please select at least one year to analyze.")
            else:
                with st.spinner("Running hiring difficulties analysis pipeline... This may take several minutes."):
                    system = initialize_system()
                    if system:
                        try:
                            # This method is assumed to be in FinancialVectorRAG
                            results = system.run_hiring_difficulties_analysis_pipeline(
                                years=years,
                                limit_companies=limit_companies
                            )
                            display_hiring_pipeline_results(results)
                        except AttributeError:
                            st.error("The analysis pipeline method is not implemented in the backend.")
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
                    else:
                        st.error("Failed to initialize system.")


def display_hiring_pipeline_results(results: Dict):
    """Display hiring difficulties pipeline analysis results."""
    if 'errors' in results and results.get('errors'):
        st.warning("⚠️ Pipeline completed with some errors.")
        with st.expander("View Errors"):
            for error in results['errors']:
                st.error(error)
    else:
        st.success("✅ Pipeline completed successfully!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filings Analyzed", results.get('filings_analyzed', 0))
    with col2:
        st.metric("Companies Analyzed", results.get('companies_analyzed', 0))

    if results.get('output_file'):
        st.info(f"Results saved to: `{results['output_file']}`")
        try:
            with open(results['output_file'], 'rb') as f:
                st.download_button(
                    "📥 Download Results CSV",
                    data=f,
                    file_name=os.path.basename(results['output_file']),
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not read output file for download: {e}")


def display_hiring_statistics():
    """Display hiring difficulties analysis statistics"""
    st.subheader("📊 Hiring Difficulties Analysis Statistics")
    
    try:
        system = initialize_system()
        if not system:
            st.error("Failed to initialize system")
            return
        
        stats = system.get_hiring_difficulties_summary_stats()
        
        if 'error' in stats:
            st.error(f"Failed to get statistics: {stats['error']}")
            return
        
        # Overview metrics
        st.subheader("📈 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Companies", stats.get('total_companies', 0))
        
        with col2:
            st.metric("Total Filings", stats.get('total_filings', 0))
        
        with col3:
            st.metric("Companies with Difficulties", stats.get('companies_with_difficulties', 0))
        
        with col4:
            st.metric("Significant Findings", stats.get('significant_findings', 0))
        
        # Average scores
        if 'avg_hiring_difficulty_score' in stats:
            st.subheader("🎯 Average Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Difficulty Score", f"{stats['avg_hiring_difficulty_score']:.2f}")
            
            with col2:
                st.metric("Average Likelihood", f"{stats['avg_hiring_difficulty_likelihood']:.1%}")
        
        # Top difficulty terms
        if 'top_difficulty_terms' in stats and stats['top_difficulty_terms']:
            st.subheader("🔍 Most Common Hiring Difficulty Indicators")
            
            terms_data = []
            for term, count in stats['top_difficulty_terms'][:15]:
                terms_data.append({'Term': term, 'Mentions': count})
            
            if terms_data:
                df_terms = pd.DataFrame(terms_data)
                st.dataframe(df_terms, use_container_width=True)
        
        # Years covered
        if 'years_covered' in stats and stats['years_covered']:
            st.subheader("📅 Analysis Coverage")
            years = sorted(stats['years_covered'])
            st.write(f"**Years covered:** {', '.join(map(str, years))}")
            st.write(f"**Time span:** {years[0]} - {years[-1]} ({len(years)} years)")
        
    except Exception as e:
        st.error(f"Error displaying statistics: {e}")


def main():
    """Main application"""
    st.sidebar.title("🏦 Financial Graph RAG")
    st.sidebar.markdown("---")

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Vector Store Overview", "M&A Query Interface", "Skill Shortage Analysis", "Hiring Difficulties Analysis", "Trends & Analytics", "System Dashboard"],
        icons=["database", "search", "people", "person-exclamation", "graph-up", "speedometer2"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # System status in sidebar
    st.sidebar.subheader("📊 System Status")
    status = get_system_status()

    if status:
        st.sidebar.success("✅ System Online")

        vector_status = status.get('vector_store', {})
        st.sidebar.metric("Collections",
                          vector_status.get('company_collections', 0))
        st.sidebar.metric("Total Documents",
                          vector_status.get('total_documents', 0))

        sp500_status = status.get('sp500_data', {})
        st.sidebar.metric("S&P 500 Companies",
                          sp500_status.get('companies_loaded', 0))
    else:
        st.sidebar.error("❌ System Offline")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    
    if selected == "M&A Query Interface":
        st.sidebar.markdown("""
        **Query Types:**
        - **Company-Specific**: Focus on one company
        - **Sector Analysis**: Analyze entire sectors
        - **Custom Selection**: Compare specific companies
        - **Cross-Company**: Search across all companies
        
        **Sample M&A Queries:**
        - "cultural integration challenges"
        - "organizational restructuring"
        - "synergies cost savings"
        - "post-merger integration"
        - "leadership changes acquisition"
        """)
    elif selected == "Skill Shortage Analysis":
        st.sidebar.markdown("""
        ### 💡 Tips for Skill Shortage Analysis
        
        **Analysis Types:**
        - **Company Analysis**: Analyze individual companies for skill shortage indicators
        - **Company Rankings**: Rank all companies by skill shortage similarity
        - **Sector Comparison**: Compare skill shortage levels across sectors
        - **Trend Analysis**: Track skill shortage patterns over time
        - **Pipeline Analysis**: Analyze hiring and training pipeline gaps
        
        **Sample Focus Areas:**
        - "technology skills"
        - "software engineers"
        - "healthcare professionals"
        - "manufacturing workers"
        - "data scientists"
        - "cybersecurity experts"
        
        **Company Rankings:**
        - Uses similarity-based analysis combining AI likelihood scores and keyword patterns
        - Ranks companies from highest to lowest skill shortage indicators
        - Includes filtering options for recent years and minimum filings
        - Provides comprehensive metrics and exportable reports
        
        **Best Practices:**
        - Use specific skill areas for more targeted analysis
        - Enable trend analysis for historical context
        - Compare results across multiple companies/sectors
        - Review AI summaries for detailed insights
        """, unsafe_allow_html=True)
    elif selected == "Hiring Difficulties Analysis":
        st.sidebar.markdown("""
        ### 💡 Tips for Hiring Difficulties Analysis
        
        **Analysis Types:**
        - **Company Analysis**: Analyze individual companies for hiring challenges for specific years.
        - **Run Full Pipeline**: Run the analysis for all S&P 500 companies for selected years to generate a summary file.
        
        **Best Practices:**
        - To analyze a single company quickly, use the "Company Analysis" tab.
        - To perform a bulk analysis, use the "Run Full Pipeline" tab. This can take a long time.
        - The results indicate whether hiring difficulties were mentioned and provide an AI-generated summary.
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        **Navigation:**
        - **Vector Store**: View data overview
        - **M&A Queries**: Search M&A impacts
        - **Skill Analysis**: Analyze skill shortages
        - **Hiring Analysis**: Analyze hiring difficulties
        - **Trends**: View analytics and trends
        - **Dashboard**: System status and metrics
        """)

    # Main content
    if selected == "Vector Store Overview":
        display_company_overview()
    elif selected == "M&A Query Interface":
        display_query_interface()
    elif selected == "Skill Shortage Analysis":
        display_skill_shortage_analysis()
    elif selected == "Hiring Difficulties Analysis":
        display_hiring_difficulties_analysis()
    elif selected == "Trends & Analytics":
        display_trends_analytics()
    elif selected == "System Dashboard":
        display_system_dashboard()


if __name__ == "__main__":
    main()
