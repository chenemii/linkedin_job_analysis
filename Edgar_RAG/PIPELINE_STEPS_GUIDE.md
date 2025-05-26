# Financial Vector RAG Pipeline Steps Guide

This guide provides the exact command line steps to download 10-K filings, store them in vector stores, and analyze M&A activity using the Financial Vector RAG system.

## Overview

The Financial Vector RAG system follows this pipeline:
1. **Download 10-K filings** from SEC EDGAR database
2. **Store documents** in company-specific vector store collections (ChromaDB)
3. **Analyze M&A activity** using RAG-based queries

## Prerequisites

Before running the pipeline, ensure you have:

1. **Python environment** with dependencies installed:
   ```bash
   cd financial_graph_rag
   pip install -r requirements.txt
   ```

2. **Environment configuration**:
   ```bash
   # Copy example environment file
   cp env_example.txt .env
   
   # Edit .env with your settings (API keys, etc.)
   # For local Ollama (default):
   # OPENAI_API_KEY=ollama-local
   # OPENAI_BASE_URL=http://localhost:11434/v1
   ```

3. **LLM setup** (using Ollama locally - recommended):
   ```bash
   # Install and start Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull gemma2:27b
   ollama serve
   ```

## Step-by-Step Pipeline Commands

### Step 1: Complete Pipeline Setup (One Command)

The simplest way to run the entire pipeline:

```bash
# Download 10-K filings and store in vector stores for multiple companies
cd financial_graph_rag
python -m financial_graph_rag.cli setup --years 2023 --limit-companies 10 --limit-filings 2
```

**What this does:**
- ✅ Collects S&P 500 company information
- ✅ Downloads 10-K filings from EDGAR for 2023
- ✅ Filters for M&A relevant content
- ✅ Stores documents in company-specific vector store collections
- ✅ Creates ChromaDB collections (e.g., "company_msft", "company_aapl")

**Output example:**
```
🚀 Setting up Financial Vector RAG data pipeline...
Years: [2023]
Company limit: 10
Filings per company: 2

✅ Data pipeline setup completed!
📊 Companies collected: 10
📄 Filings downloaded: 18
📁 Documents stored: 15
🗃️ Company collections created: 8
```

### Step 2: Check System Status

Verify that documents were successfully stored:

```bash
python -m financial_graph_rag.cli status
```

**Output example:**
```
📊 System Status: operational

🗃️ Vector Store:
  Company collections: 8
  Total documents: 15
  Companies with data: MSFT, AAPL, GOOGL, AMZN, TSLA, NVDA, META, NFLX

📈 S&P 500 Data:
  Companies loaded: 503
  Companies with CIK: 487

📄 EDGAR Data:
  Filings cached: 18
```

### Step 3: Analyze M&A Activity for Specific Companies

#### 3.1 Basic M&A Analysis

```bash
# Analyze Microsoft's M&A organizational impact
python -m financial_graph_rag.cli analyze MSFT --focus "organizational structure changes"
```

**Output example:**
```
🔍 Analyzing M&A impact for MSFT...
Focus: organizational structure changes

🏢 Company: Microsoft Corporation (MSFT)
📍 Sector: Information Technology  
🏛️ Headquarters: Redmond, Washington

📊 Analysis Summary:
Documents analyzed: 2

M&A Activity Summary:
- Microsoft completed acquisition of Activision Blizzard for $68.7B in 2023
- Strategic focus on gaming and metaverse capabilities...

Organizational Structure Impact:
- Created new Gaming division under CEO Phil Spencer
- Integrated 10,000+ employees from Activision...
```

#### 3.2 Focused M&A Analysis with Output

```bash
# Analyze with specific focus and save results
python -m financial_graph_rag.cli analyze AAPL \
  --focus "acquisition integration strategies" \
  --output apple_ma_analysis.txt \
  --format txt
```

#### 3.3 Multiple Company Analysis

```bash
# Analyze different companies for different aspects
python -m financial_graph_rag.cli analyze GOOGL --focus "subsidiary restructuring"
python -m financial_graph_rag.cli analyze AMZN --focus "vertical integration through acquisitions"
python -m financial_graph_rag.cli analyze TSLA --focus "technology acquisition impact"
```

### Step 4: Advanced M&A Analysis

#### 4.1 Sector-Wide M&A Trends

```bash
# Analyze M&A trends across Technology sector
python -m financial_graph_rag.cli trends --sector "Information Technology" --output tech_ma_trends.json
```

#### 4.2 Cross-Company M&A Comparison

```bash
# Compare specific companies' M&A patterns
python -m financial_graph_rag.cli trends --companies MSFT AAPL GOOGL --output big_tech_ma.json
```

#### 4.3 Find Similar M&A Patterns

```bash
# Find companies with similar M&A patterns to Microsoft
python -m financial_graph_rag.cli similar MSFT --criteria "integration challenges" --output similar_to_msft.json
```

### Step 5: Custom Queries

#### 5.1 Company-Specific Queries

```bash
# Search Microsoft's documents for specific M&A topics
python -m financial_graph_rag.cli query "post-merger integration challenges" --company MSFT

# Search for synergy discussions
python -m financial_graph_rag.cli query "synergies cost savings merger" --company AAPL

# Search for organizational restructuring
python -m financial_graph_rag.cli query "organizational restructuring divisions" --company GOOGL
```

#### 5.2 Cross-Company Queries

```bash
# Search across all companies for M&A trends
python -m financial_graph_rag.cli query "cultural integration merger acquisition"

# Search for specific M&A strategies
python -m financial_graph_rag.cli query "horizontal vertical integration acquisition strategy"
```

### Step 6: Get Detailed Company Information

```bash
# Get detailed information about a company's data
python -m financial_graph_rag.cli company-info MSFT
```

**Output example:**
```
🏢 Getting information for MSFT...

📊 Company Information:
  Name: Microsoft Corporation
  Sector: Information Technology
  Headquarters: Redmond, Washington
  CIK: 0000789019

🗃️ Vector Store Statistics:
  Document count: 2
  Status: active
  Form types: 10-K
  Average M&A score: 0.75
```

## Pipeline Commands for Different Scenarios

### Scenario 1: Quick Test with Single Company

```bash
# Test pipeline with just one company
python -m financial_graph_rag.cli setup --years 2023 --limit-companies 1 --limit-filings 1

# Check what was stored
python -m financial_graph_rag.cli status

# Analyze the company
python -m financial_graph_rag.cli company-info MSFT
python -m financial_graph_rag.cli analyze MSFT
```

### Scenario 2: Comprehensive Multi-Year Analysis

```bash
# Download multiple years of data for comprehensive analysis
python -m financial_graph_rag.cli setup --years 2022 --years 2023 --limit-companies 20

# Analyze trends over time
python -m financial_graph_rag.cli trends --companies MSFT AAPL GOOGL AMZN

# Detailed analysis for specific companies
python -m financial_graph_rag.cli analyze MSFT --focus "multi-year organizational evolution"
```

### Scenario 3: Sector-Specific Research

```bash
# Focus on specific sector
python -m financial_graph_rag.cli setup --years 2023 --limit-companies 50

# Filter and analyze by sector
python -m financial_graph_rag.cli trends --sector "Information Technology"
python -m financial_graph_rag.cli trends --sector "Health Care"
python -m financial_graph_rag.cli trends --sector "Financials"
```

## Using the Standalone Pipeline Script

For more detailed control, you can also use the `ma_analysis_pipeline.py` script:

```bash
# Analyze single company with detailed output
python ma_analysis_pipeline.py --ticker MSFT --year 2023 --detailed-analysis

# Analyze multiple years
python ma_analysis_pipeline.py --ticker AAPL --years 2022 2023 --save-report

# Verbose output for debugging
python ma_analysis_pipeline.py --ticker GOOGL --year 2023 --verbose
```

## Testing the Pipeline

Run the test script to verify everything works:

```bash
python test_pipeline_steps.py
```

This will test:
- ✅ S&P 500 company collection
- ✅ 10-K filing download from EDGAR
- ✅ M&A content filtering  
- ✅ Vector store storage (company-specific collections)
- ✅ Document retrieval from vector store
- ✅ M&A analysis using RAG

## Output Files

The system creates several types of output:

### Analysis Reports
```
reports/
├── ma_analysis_MSFT_2023_20241201_143022.json    # Detailed JSON results
├── ma_analysis_MSFT_2023_20241201_143022.txt     # Human-readable report
├── tech_ma_trends.json                           # Sector analysis
└── similar_to_msft.json                          # Similarity analysis
```

### Vector Store Data
```
chroma_db/
├── company_msft/         # Microsoft's documents
├── company_aapl/         # Apple's documents  
├── company_googl/        # Google's documents
└── ...
```

### Cache Files
```
cache/
├── sp500_companies.json  # S&P 500 company data
└── edgar_filings/        # Downloaded 10-K filings
```

## Key Pipeline Features

1. **Company-Specific Collections**: Each company gets its own vector store collection to prevent cross-contamination
2. **M&A Content Filtering**: Automatically identifies and prioritizes filings with M&A content
3. **Intelligent Chunking**: Breaks down long 10-K filings into manageable, searchable chunks
4. **Metadata Preservation**: Maintains filing dates, form types, and M&A relevance scores
5. **Incremental Updates**: Avoids re-downloading existing filings
6. **Error Handling**: Continues processing even if some filings fail to download

## Troubleshooting

### Common Issues

1. **No documents found**: Check if filings were downloaded successfully
   ```bash
   python -m financial_graph_rag.cli status
   ```

2. **API rate limits**: EDGAR has rate limits; the system handles this automatically with delays

3. **Memory issues**: For large datasets, use `--limit-companies` and `--limit-filings` options

4. **LLM errors**: Ensure Ollama is running or API keys are configured correctly

5. **Query results not showing**: This issue has been fixed in the latest version. The CLI now properly displays cross-company query results with document content.

### Debug Commands

```bash
# Verbose logging
python -m financial_graph_rag.cli setup --years 2023 --limit-companies 1 --verbose

# Check specific company data
python -m financial_graph_rag.cli company-info MSFT

# Test with simple query
python -m financial_graph_rag.cli query "test" --company MSFT

# Run debug script
python debug_query_issue.py
```

### Working Query Examples

The query command now works correctly and displays detailed results:

```bash
# Cross-company search (shows results from all companies)
python -m financial_graph_rag.cli query "merger acquisition"
python -m financial_graph_rag.cli query "cultural integration merger acquisition" 
python -m financial_graph_rag.cli query "organizational restructuring"

# Company-specific search
python -m financial_graph_rag.cli query "merger acquisition" --company ABT
python -m financial_graph_rag.cli query "synergies cost savings" --company MSFT
```

**Expected output format:**
```
📊 Query Results:
Query: merger acquisition
Companies searched: 10
Companies with results: 10
Total documents: 10

🏢 Results by Company:

📈 ABT: 1 documents
  📄 Document 1 (Score: 1.756):
    [Document content preview...]

📈 ADBE: 1 documents
  📄 Document 1 (Score: 1.702):
    [Document content preview...]
```

This comprehensive guide covers all the command line steps needed to download 10-K filings, store them in vector stores, and analyze M&A activity using the Financial Vector RAG system. 