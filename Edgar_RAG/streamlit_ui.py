"""
Streamlit UI for Financial Vector RAG System

A comprehensive web interface for viewing vector store contents and querying
the Financial Vector RAG system for M&A analysis.
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
st.set_page_config(page_title="Financial Vector RAG Explorer",
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


def main():
    """Main application"""
    st.sidebar.title("🏦 Financial Vector RAG")
    st.sidebar.markdown("---")

    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Vector Store Overview", "Query Interface"],
        icons=["database", "search"],
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
    st.sidebar.markdown("""
    **Query Types:**
    - **Company-Specific**: Focus on one company
    - **Sector Analysis**: Analyze entire sectors
    - **Custom Selection**: Compare specific companies
    - **Cross-Company**: Search across all companies
    
    **Sample Queries:**
    - "cultural integration challenges"
    - "organizational restructuring"
    - "synergies cost savings"
    - "post-merger integration"
    - "leadership changes acquisition"
    """)

    # Main content
    if selected == "Vector Store Overview":
        display_company_overview()
    elif selected == "Query Interface":
        display_query_interface()


if __name__ == "__main__":
    main()
