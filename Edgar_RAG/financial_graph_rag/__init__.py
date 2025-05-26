"""
Financial Vector RAG - M&A Organizational Structure Analysis

A comprehensive system for analyzing M&A impacts on organizational structure using:
- SEC EDGAR 10-K filing collection and processing
- Vector-based document storage with company-specific collections
- Retrieval Augmented Generation (RAG) for intelligent analysis
- Focus on M&A organizational changes and integration patterns

Key Features:
- Company-specific vector stores for better precision
- Advanced M&A content filtering and scoring
- Temporal analysis of organizational changes
- Cross-company trend analysis and benchmarking
"""

from .core import FinancialVectorRAG
from .rag_engine import VectorRAGEngine
from .config import settings

__version__ = "2.0.0"
__author__ = "Emily Chen"

__all__ = ["FinancialVectorRAG", "VectorRAGEngine", "settings"]
