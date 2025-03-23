# LinkedIn Job Posting Analysis

This repository provides tools for analyzing LinkedIn job postings to extract insights about job requirements, skills, trends, and organizational structures.

## Overview

This toolkit allows you to analyze LinkedIn job posting data to gain insights into:
- Job level distributions (Entry, Mid, Senior)
- Salary trends across different job levels
- Common skills and requirements
- Organizational structures inferred from job descriptions
- Key terms used in different types of organizations

## Quick Start

The easiest way to analyze LinkedIn job postings is to use the provided script:

```bash
# Run the analysis script on your data
python linkedin_job_analysis.py
```

The script will:
1. Load LinkedIn job posting data from `postings.csv` or other compatible files
2. Clean and preprocess the job descriptions
3. Analyze job levels, skills, salaries, and organizational structures
4. Generate visualizations as PNG files
5. Create an HTML report with all visualizations and insights

## Visualizations Generated

The script generates several visualizations:

- **job_levels.png**: Distribution of job levels (Entry, Mid, Senior)
- **job_terms_wordcloud.png**: Word cloud of most common terms in job descriptions
- **salary_by_level.png**: Box plot of salary distributions by job level
- **org_structures.png**: Distribution of organizational structures (Flat, Hierarchical, Hybrid)
- **org_structure_by_level.png**: Job levels by organizational structure
- **top_skills.png**: Top skills mentioned in job postings

All these visualizations are compiled into an **HTML report** (`visualization_results.html`) that can be viewed in any browser.

## Setup Instructions

### Option 1: Using the Analysis Script (Recommended)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data (should be named `postings.csv` or one of the supported formats)

4. Run the analysis script:
   ```bash
   python linkedin_job_analysis.py
   ```

5. View the results:
   ```bash
   open visualization_results.html  # On macOS
   # Or open the file directly in your browser
   ```

## Analysis Features

- **Job Level Analysis**: Identify and visualize entry, mid, and senior level positions
- **Skills Analysis**: Extract and analyze common skills and requirements
- **Salary Analysis**: Analyze compensation trends across job levels
- **Organizational Structure Analysis**: Classify job postings into flat, hierarchical, or hybrid structures
- **Keyword Analysis**: Extract characteristic terms used in different types of organizations

