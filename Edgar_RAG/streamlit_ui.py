"""
Streamlit UI for Financial Vector RAG System

A comprehensive web interface for viewing vector store contents and querying
the Financial Vector RAG system for M&A analysis and skill shortage analysis.
"""

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


@st.cache_resource
def initialize_system():
    """Initialize the Financial Vector RAG system"""
    try:
        system = FinancialVectorRAG()
        return system
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Company Analysis", "Sector Comparison", "Trend Analysis", "Pipeline Analysis", "CSV Analysis"]
    )
    
    if analysis_type == "Company Analysis":
        display_company_skill_analysis()
    elif analysis_type == "Sector Comparison":
        display_sector_skill_comparison()
    elif analysis_type == "Trend Analysis":
        display_skill_trend_analysis()
    elif analysis_type == "Pipeline Analysis":
        display_skill_pipeline_analysis()
    elif analysis_type == "CSV Analysis":
        display_csv_skill_analysis()


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
            with st.spinner(f"Analyzing skill shortages for {company_ticker}..."):
                system = initialize_system()
                if system:
                    try:
                        results = system.analyze_company_skill_shortage(
                            company_ticker=company_ticker,
                            focus_area=focus_area if focus_area else None
                        )
                        
                        if 'error' not in results:
                            display_company_skill_results(results, company_ticker)
                        else:
                            st.error(f"Analysis failed: {results['error']}")
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                else:
                    st.error("System not available")


def display_company_skill_results(results: Dict, company_ticker: str):
    """Display company skill shortage analysis results"""
    st.success(f"✅ Analysis completed for {company_ticker}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Analyzed", results.get('documents_analyzed', 0))
    with col2:
        st.metric("Skill Shortage Mentions", results.get('skill_shortage_mentions', 0))
    with col3:
        severity_score = results.get('severity_score', 0)
        st.metric("Severity Score", f"{severity_score:.1f}/10")
    with col4:
        risk_level = "High" if severity_score > 7 else "Medium" if severity_score > 4 else "Low"
        st.metric("Risk Level", risk_level)
    
    # Skill gaps
    if 'skill_gaps' in results and results['skill_gaps']:
        st.subheader("🎯 Identified Skill Gaps")
        for i, gap in enumerate(results['skill_gaps'][:10], 1):
            st.markdown(f'<div class="skill-gap-card">{i}. {gap}</div>', unsafe_allow_html=True)
    
    # Detailed analysis
    if 'analysis' in results:
        st.subheader("📊 Detailed Analysis")
        st.markdown(f'<div class="skill-shortage-result">{results["analysis"]}</div>', unsafe_allow_html=True)
    
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
                        companies=company_tickers,
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
                            company_limit=company_limit
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
    """Generate a comprehensive skill shortage report"""
    report = f"""
# Skill Shortage Analysis Report
## Company: {company_ticker}
## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Executive Summary
- Documents Analyzed: {results.get('documents_analyzed', 0)}
- Skill Shortage Mentions: {results.get('skill_shortage_mentions', 0)}
- Severity Score: {results.get('severity_score', 0):.1f}/10

### Key Findings
"""
    
    if 'skill_gaps' in results:
        report += "\n#### Identified Skill Gaps:\n"
        for i, gap in enumerate(results['skill_gaps'][:10], 1):
            report += f"{i}. {gap}\n"
    
    if 'analysis' in results:
        report += f"\n#### Detailed Analysis:\n{results['analysis']}\n"
    
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"{company_ticker}_skill_shortage_report.md",
        mime="text/markdown"
    )


def main():
    """Main application"""
    st.sidebar.title("🏦 Financial Graph RAG")
    st.sidebar.markdown("---")

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Vector Store Overview", "M&A Query Interface", "Skill Shortage Analysis", "Trends & Analytics", "System Dashboard"],
        icons=["database", "search", "people", "graph-up", "speedometer2"],
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
        **Analysis Types:**
        - **Company Analysis**: Skill gaps for specific companies
        - **Sector Comparison**: Compare skill shortages across sectors
        - **Trend Analysis**: Track skill shortage patterns over time
        - **Pipeline Analysis**: Comprehensive skill shortage review
        
        **Sample Focus Areas:**
        - "technology skills"
        - "healthcare professionals"
        - "data science talent"
        - "cybersecurity expertise"
        - "digital transformation skills"
        """)
    else:
        st.sidebar.markdown("""
        **Navigation:**
        - **Vector Store**: View data overview
        - **M&A Queries**: Search M&A impacts
        - **Skill Analysis**: Analyze skill shortages
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
    elif selected == "Trends & Analytics":
        display_trends_analytics()
    elif selected == "System Dashboard":
        display_system_dashboard()


if __name__ == "__main__":
    main()
