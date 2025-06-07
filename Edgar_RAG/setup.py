#!/usr/bin/env python3
"""
Setup script for Financial Graph RAG
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_financial_graph_rag.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Financial Graph RAG - Analyze M&A organizational impacts and skill shortages in financial filings"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'financial_graph_rag', 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="financial-graph-rag",
    version="1.0.0",
    description="Analyze M&A organizational impacts and skill shortages in financial filings",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Financial RAG Team",
    author_email="",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'financial-rag=financial_graph_rag.cli:cli',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    keywords="financial analysis, RAG, M&A, skill shortage, SEC filings, EDGAR",
) 