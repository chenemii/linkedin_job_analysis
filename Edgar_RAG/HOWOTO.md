## Installation & Setup

First, make sure you have the system installed and configured. The CLI provides a comprehensive interface for all functionality.

## Basic Usage



### 1. **Setup the Data Pipeline** (First Time)
```bash
# Setup with default years (2022, 2023)
python3 -m financial_graph_rag.cli setup

# Setup for specific years with company limit
python3 -m financial_graph_rag.cli setup --years 2020 2021 2022 --limit 50

# Force refresh of existing data
python3 -m financial_graph_rag.cli setup --years 2023 --force
```

### 2. **Check System Status**
```bash
# View system statistics and data availability
python3 -m financial_graph_rag.cli status
```

## M&A Organizational Impact Analysis

### 3. **Analyze M&A Impacts**
```bash
# Analyze all available data
python3 -m financial_graph_rag.cli analyze

# Analyze specific years with output file
python3 -m financial_graph_rag.cli analyze --years 2021 2022 --output ma_analysis.json

# Limit analysis to specific number of companies
python3 -m financial_graph_rag.cli analyze --limit 100 --output results.json
```

### 4. **Analyze Trends**
```bash
# View trends across all years
python3 -m financial_graph_rag.cli trends

# Focus on specific sector
python3 -m financial_graph_rag.cli trends --sector "Information Technology"

# Analyze trends for specific years
python3 -m financial_graph_rag.cli trends --years 2020 2021 2022 --output trends.json
```

### 5. **Find Similar Companies**
```bash
# Find companies similar to Apple
python3 -m financial_graph_rag.cli similar AAPL

# Use specific criteria for similarity
python3 -m financial_graph_rag.cli similar MSFT --criteria "organizational restructuring"

# Save results to file
python3 -m financial_graph_rag.cli similar GOOGL --output similar_companies.json
```

### 6. **Query the System**
```bash
# Natural language queries
python3 -m financial_graph_rag.cli query "workforce reduction after merger"

# Limit number of results
python3 -m financial_graph_rag.cli query "organizational restructuring" --limit 10

# Include additional context
python3 -m financial_graph_rag.cli query "acquisition integration challenges" --context
```

### 7. **Get Company Information**
```bash
# Basic company info
python3 -m financial_graph_rag.cli company-info AAPL

# Detailed information
python3 -m financial_graph_rag.cli company-info MSFT --detailed
```

## Skill Shortage Analysis

### 8. **Run Complete Skill Shortage Analysis**
```bash
# Full pipeline analysis
python3 -m financial_graph_rag.cli skill-shortage-pipeline

# Analyze specific years
python3 -m financial_graph_rag.cli skill-shortage-pipeline --years 2020 2021 2022

# Limit companies and save results
python3 -m financial_graph_rag.cli skill-shortage-pipeline --limit 50 --output skill_analysis.json
```

### 9. **Analyze Specific Company's Skill Shortages**
```bash
# Basic company skill shortage analysis
python3 -m financial_graph_rag.cli skill-shortage-company AAPL

# Focus on specific area
python3 -m financial_graph_rag.cli skill-shortage-company MSFT --focus "technology skills"

# Save in different formats
python3 -m financial_graph_rag.cli skill-shortage-company GOOGL --output results.csv --format csv
```

### 10. **Compare Skill Shortages Across Companies**
```bash
# Compare specific companies
python3 -m financial_graph_rag.cli skill-shortage-compare --companies AAPL MSFT GOOGL

# Compare within a sector
python3 -m financial_graph_rag.cli skill-shortage-compare --sector "Technology"

# Compare all companies
python3 -m financial_graph_rag.cli skill-shortage-compare --output comparison.json
```

### 11. **Analyze Skill Shortage Trends**
```bash
# Trends across all years
python3 -m financial_graph_rag.cli skill-shortage-trends

# Focus on specific years and sector
python3 -m financial_graph_rag.cli skill-shortage-trends --years 2020 2021 --sector "Healthcare"

# Save trend analysis
python3 -m financial_graph_rag.cli skill-shortage-trends --output trends.json
```

### 12. **Analyze from CSV Data**
```bash
# Analyze from your own CSV file (with columns: cik, Year, FName, gvkey)
python3 -m financial_graph_rag.cli analyze-csv data/filings.csv

# Limit processing and specify output
python3 -m financial_graph_rag.cli analyze-csv data/filings.csv --limit 100 --output csv_results.json
```

### 13. **Get Skill Shortage Statistics**
```bash
# View summary statistics
python3 -m financial_graph_rag.cli skill-shortage-stats
```

## Utility Commands

### 14. **Clean Up System**
```bash
# Clean with confirmation prompt
python3 -m financial_graph_rag.cli clean

# Clean without confirmation
python3 -m financial_graph_rag.cli clean --confirm
```

### 15. **View Usage Examples**
```bash
# See all available examples
python3 -m financial_graph_rag.cli examples
```

## Typical Workflows

### **Workflow 1: First-time Setup and Basic Analysis**
```bash
# 1. Setup the system
python3 -m financial_graph_rag.cli setup --years 2022 2023 --limit 20

# 2. Check status
python3 -m financial_graph_rag.cli status

# 3. Run basic M&A analysis
python3 -m financial_graph_rag.cli analyze --output initial_analysis.json

# 4. View trends
python3 -m financial_graph_rag.cli trends --output trends.json
```

### **Workflow 2: Skill Shortage Research**
```bash
# 1. Run complete skill shortage analysis
python3 -m financial_graph_rag.cli skill-shortage-pipeline --years 2021 2022

# 2. Analyze specific companies
python3 -m financial_graph_rag.cli skill-shortage-company AAPL --output apple_skills.json
python3 -m financial_graph_rag.cli skill-shortage-company MSFT --output microsoft_skills.json

# 3. Compare across sector
python3 -m financial_graph_rag.cli skill-shortage-compare --sector "Technology" --output tech_comparison.json

# 4. View trends
python3 -m financial_graph_rag.cli skill-shortage-trends --output skill_trends.json
```

### **Workflow 3: Custom Research Queries**
```bash
# Query specific topics
python3 -m financial_graph_rag.cli query "remote work policies after acquisition"
python3 -m financial_graph_rag.cli query "talent retention strategies during merger"
python3 -m financial_graph_rag.cli query "organizational culture integration"

# Find similar patterns
python3 -m financial_graph_rag.cli similar AAPL --criteria "workforce management"
```

## Configuration

The system uses configuration settings from `financial_graph_rag/config.py`. Key settings include:
- Data directories
- EDGAR API settings
- Vector store configuration
- LLM settings

## Output Formats

Most commands support different output formats:
- **JSON**: Structured data (default)
- **CSV**: Tabular data
- **TXT**: Human-readable reports

