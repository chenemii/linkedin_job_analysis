# EDGAR 10-K Filing Downloader

This document describes how to download 10-K filings from the SEC EDGAR database using the Financial Graph RAG system.

## Overview

The system provides two ways to download EDGAR 10-K filings:

1. **CLI Command**: `download-filings` - Integrated into the main CLI
2. **Standalone Script**: `download_edgar_filings.py` - Can be used independently

Both methods support:
- ✅ **Specific Company Selection**: Download by ticker symbols
- ✅ **Year Filtering**: Choose specific years 
- ✅ **M&A Content Detection**: Automatically identify M&A relevant filings
- ✅ **Vector Store Integration**: Add documents to RAG system
- ✅ **Progress Tracking**: Real-time download progress
- ✅ **Error Handling**: Robust error handling and reporting
- ✅ **Caching**: Save results for future use

## Entity Extraction Methods

The system supports two methods for extracting entities and relationships from 10-K filings:

### 1. LLM Method (Default)
- **Usage**: `--extraction-method llm`
- **Description**: Uses general-purpose LLMs (Ollama/OpenAI) for extraction
- **Pros**: More accurate, handles complex scenarios well
- **Cons**: Slower, requires more computational resources
- **Best for**: High-accuracy extraction, complex document analysis

### 2. Triplex Method (Recommended for Speed)
- **Usage**: `--extraction-method triplex`
- **Description**: Uses the specialized [SciPhi Triplex model](https://huggingface.co/SciPhi/Triplex) designed for knowledge graph construction
- **Pros**: Much faster (98% cost reduction vs GPT-4), runs locally, specialized for knowledge graphs
- **Cons**: May be less accurate for very complex relationships
- **Best for**: Fast extraction, large-scale processing, local deployment

### Installation for Triplex

To use the Triplex extraction method, install additional dependencies:

```bash
# Install Triplex dependencies
pip install -r requirements-triplex.txt

# Or install manually:
pip install transformers torch accelerate safetensors
```

### Triplex Usage Examples

```bash
# Fast setup with Triplex
python -m financial_graph_rag.cli setup \
  --years 2023 \
  --limit-companies 50 \
  --extraction-method triplex

# Download and extract with Triplex
python -m financial_graph_rag.cli download-filings \
  --tickers AAPL MSFT GOOGL \
  --years 2023 \
  --add-to-vector-store \
  --extraction-method triplex
```

## Quick Start

### 1. Download filings for specific companies
```bash
# Download recent 10-K filings for Apple, Microsoft, and Google
python -m financial_graph_rag.cli download-filings \
  --tickers AAPL MSFT GOOGL \
  --years 2023 \
  --add-to-vector-store
```

### 2. Download for analysis pipeline (Fast with Triplex)
```bash
# Download M&A relevant filings and extract with Triplex (much faster)
python -m financial_graph_rag.cli download-filings \
  --years 2022 2023 \
  --limit-companies 10 \
  --ma-only \
  --add-to-vector-store \
  --extraction-method triplex
```

### 3. Bulk download with limits
```bash
# Download recent filings for top 50 companies (1 filing each)
python -m financial_graph_rag.cli download-filings \
  --years 2023 \
  --limit-companies 50 \
  --limit-filings 1
```

## CLI Command Reference

### Basic Usage
```bash
python -m financial_graph_rag.cli download-filings [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--years` | `-y` | Years of filings to download | 2023 |
| `--tickers` | `-t` | Specific company tickers | All S&P 500 |
| `--limit-companies` | `-c` | Max companies to process | None |
| `--limit-filings` | `-f` | Max filings per company | 1 |
| `--add-to-vector-store` | | Add to vector store for RAG | False |
| `--ma-only` | | Only keep M&A relevant filings | False |
| `--extraction-method` | | Extraction method | llm |

### Examples

#### Target Specific Companies
```bash
# Tech giants
python -m financial_graph_rag.cli download-filings \
  --tickers AAPL MSFT GOOGL AMZN \
  --years 2023

# Financial sector  
python -m financial_graph_rag.cli download-filings \
  --tickers JPM BAC WFC GS \
  --years 2022 2023 \
  --limit-filings 2
```

#### Bulk Downloads
```bash
# Recent filings for analysis
python -m financial_graph_rag.cli download-filings \
  --years 2023 \
  --limit-companies 25 \
  --add-to-vector-store

# M&A focused dataset
python -m financial_graph_rag.cli download-filings \
  --years 2021 2022 2023 \
  --ma-only \
  --add-to-vector-store
```

## Standalone Script Usage

The standalone script can be used without the full CLI system:

```bash
python financial_graph_rag/scripts/download_edgar_filings.py [OPTIONS]
```

### Standalone Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--years` | Years to download | 2023 |
| `--tickers` | Specific tickers | All S&P 500 |
| `--limit-companies` | Max companies | None |
| `--limit-filings` | Max filings per company | 1 |
| `--ma-only` | M&A relevant only | False |
| `--output-dir` | Custom output directory | data/edgar |
| `--report-file` | Save JSON report | None |
| `--verbose` | Enable debug logging | False |

### Standalone Examples

```bash
# Download and save report
python financial_graph_rag/scripts/download_edgar_filings.py \
  --tickers AAPL MSFT \
  --years 2023 \
  --report-file download_report.json \
  --verbose

# Custom output directory
python financial_graph_rag/scripts/download_edgar_filings.py \
  --years 2023 \
  --limit-companies 5 \
  --output-dir ./my_filings
```

## Output and Results

### File Storage
- **Location**: `data/edgar/sec-edgar-filings/`
- **Structure**: `{CIK}/10-K/{accession-number}/`
- **Files**: Original EDGAR `.txt` files

### Download Statistics
The system provides comprehensive statistics:

```
✅ Download completed!
📊 Total filings downloaded: 5
📊 Successfully processed: 5  
📊 M&A relevant filings: 3
📊 Companies covered: 5
📊 Years covered: 1
📊 M&A relevance rate: 60.0%

📈 Top companies by filing count:
  • Apple Inc.: 1 filings
  • Microsoft: 1 filings
  • Alphabet Inc. (Class A): 1 filings
```

### Caching
- **Cache File**: `cache/edgar_filings.json`
- **Purpose**: Avoid re-downloading existing filings
- **Content**: Filing metadata and M&A scores

## M&A Content Detection

The system automatically scores filings for M&A relevance using keyword analysis:

### M&A Keywords Detected
- merger, acquisition, acquire, acquired
- subsidiary, consolidation, integration
- joint venture, partnership, alliance
- divestiture, spin-off, reorganization
- And many more...

### M&A Scoring
- **Score**: Percentage of M&A-related words in document
- **Threshold**: 0.01% (1 in 10,000 words)
- **Use Cases**: Filter relevant documents, prioritize processing

## Integration with Analysis

### Vector Store Integration
```bash
# Download and add to vector store
python -m financial_graph_rag.cli download-filings \
  --tickers MSFT \
  --years 2023 \
  --add-to-vector-store

# Now analyze with document context
python -m financial_graph_rag.cli analyze MSFT \
  --focus "subsidiary integration"
```

### Verification
```bash
# Check vector store has documents
python -c "
from financial_graph_rag.rag_engine.graph_rag import GraphRAGEngine
engine = GraphRAGEngine()
print(f'Documents in store: {engine.collection.count()}')
"
```

## Best Practices

### 1. Start Small
```bash
# Test with a few companies first
python -m financial_graph_rag.cli download-filings \
  --tickers AAPL MSFT \
  --years 2023 \
  --limit-filings 1
```

### 2. Use Rate Limiting
The system automatically respects SEC fair access policies:
- ✅ 10 requests per second limit
- ✅ Proper user-agent headers
- ✅ Exponential backoff on errors

### 3. Filter for Relevance
```bash
# Focus on M&A relevant content
python -m financial_graph_rag.cli download-filings \
  --years 2023 \
  --ma-only \
  --add-to-vector-store
```

### 4. Monitor Progress
```bash
# Use verbose mode for debugging
python financial_graph_rag/scripts/download_edgar_filings.py \
  --tickers AAPL \
  --verbose
```

## Configuration

### Environment Variables
Set in your `.env` file:

```bash
# SEC EDGAR Configuration
EDGAR_USER_AGENT="YourCompanyName"
EDGAR_EMAIL="your.email@company.com"
EDGAR_DATA_DIRECTORY="data/edgar"
EDGAR_RATE_LIMIT=0.1  # Seconds between requests

# Cache Directory
CACHE_DIRECTORY="cache"
```

### Data Directories
```
data/
├── edgar/
│   └── sec-edgar-filings/
│       └── {CIK}/
│           └── 10-K/
│               └── {accession-number}/
│                   └── filing.txt
cache/
├── edgar_filings.json
└── sp500_companies.json
```

## Troubleshooting

### Common Issues

#### 1. No CIK Available
```
⚠️  Ticker XYZ not found in S&P 500
```
**Solution**: Only S&P 500 companies are supported. Check ticker symbol.

#### 2. Download Errors
```
❌ AAPL: Error - HTTP 403 Forbidden
```
**Solution**: Rate limiting activated. Wait and retry, or check SEC fair access compliance.

#### 3. Empty Results
```
📊 Total filings downloaded: 0
```
**Solution**: Check year range, company may not have filed 10-K in specified years.

### Debug Mode
```bash
# Enable verbose logging
python financial_graph_rag/scripts/download_edgar_filings.py \
  --tickers AAPL \
  --verbose
```

### Rate Limit Issues
If you encounter rate limiting:
1. Check your `EDGAR_RATE_LIMIT` setting
2. Ensure proper `EDGAR_USER_AGENT` and `EDGAR_EMAIL`
3. Consider processing smaller batches

## System Requirements

### Dependencies
- `sec-edgar-downloader>=5.0.0`
- `beautifulsoup4`
- `requests`
- `pandas`

### Storage Requirements
- **Per Filing**: ~1-5MB
- **100 Companies**: ~500MB-2.5GB
- **Full S&P 500**: ~2.5GB-12.5GB

### Network Requirements
- Stable internet connection
- Respect for SEC rate limits (10 req/sec)

## Next Steps

After downloading filings:

1. **Analyze Companies**: Use the `analyze` command
2. **Extract Entities**: Run the full `setup` pipeline  
3. **Query Knowledge**: Use the `query` command
4. **Compare Trends**: Use the `trends` command

Example full workflow:
```bash
# 1. Download filings
python -m financial_graph_rag.cli download-filings \
  --tickers AAPL MSFT GOOGL \
  --years 2023 \
  --add-to-vector-store

# 2. Analyze specific company
python -m financial_graph_rag.cli analyze MSFT \
  --focus "acquisition strategy"

# 3. Compare trends
python -m financial_graph_rag.cli trends \
  --sector "Information Technology" \
  --years 2023
``` 