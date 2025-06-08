"""
Vector Store RAG Engine

RAG engine that uses company-specific vector store collections 
for analyzing M&A impacts on organizational structure.
Each company gets its own collection to reduce noise and improve precision.
"""

import logging
from typing import List, Dict, Optional, Any, Set
import json
import re
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..config import settings

logger = logging.getLogger(__name__)


class VectorRAGEngine:
    """
    Vector Store RAG engine for financial analysis using company-specific collections
    """

    def __init__(self):
        """Initialize the Vector RAG engine"""
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Create prompts
        self.analysis_prompt = self._create_analysis_prompt()

        # Track company collections
        self._company_collections: Dict[str, chromadb.Collection] = {}

    def _initialize_llm(self):
        """Initialize the language model"""
        if settings.default_llm_provider == "openai":
            return ChatOpenAI(model=settings.default_llm_model,
                              temperature=0.1,
                              api_key=settings.openai_api_key,
                              base_url=getattr(settings, 'openai_base_url',
                                               None))
        elif settings.default_llm_provider == "anthropic":
            return ChatAnthropic(model=settings.default_llm_model,
                                 temperature=0.1,
                                 api_key=settings.anthropic_api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {settings.default_llm_provider}")

    def _normalize_company_name(self, company_name: str) -> str:
        """Normalize company name for collection naming"""
        # Remove special characters and spaces, convert to lowercase
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', company_name.lower())
        # Remove consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        # Limit length
        if len(normalized) > 50:
            normalized = normalized[:50]
        return f"company_{normalized}"

    def _get_company_collection(
            self,
            company_ticker: str,
            company_name: str = None) -> chromadb.Collection:
        """Get or create collection for a specific company"""

        # Use ticker as primary identifier, fallback to name
        collection_key = company_ticker.upper()

        if collection_key in self._company_collections:
            return self._company_collections[collection_key]

        # Create collection name
        collection_name = self._normalize_company_name(company_ticker)

        try:
            collection = self.chroma_client.get_collection(
                name=collection_name)
            logger.info(
                f"Retrieved existing collection for {company_ticker}: {collection_name}"
            )
        except:
            # Create new collection
            metadata = {
                "description": f"Financial documents for {company_ticker}",
                "company_ticker": company_ticker,
                "created_at": datetime.now().isoformat()
            }
            if company_name:
                metadata["company_name"] = company_name

            collection = self.chroma_client.create_collection(
                name=collection_name, metadata=metadata)
            logger.info(
                f"Created new collection for {company_ticker}: {collection_name}"
            )

        self._company_collections[collection_key] = collection
        return collection

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the analysis prompt template"""
        system_prompt = """You are a financial analyst specializing in M&A activities and organizational structure analysis.
        
        You will analyze SEC filings and other financial documents to provide insights about mergers, acquisitions, 
        and their impact on organizational structures.
        
        Focus on:
        - M&A transactions and their strategic rationale
        - Organizational changes following M&A
        - Integration challenges and synergies
        - Cultural and operational impacts
        - Leadership and structural reorganization
        
        Provide specific, actionable insights based on the document evidence provided."""

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
             "Based on the following documents, analyze: {query}\n\nDocument Context:\n{document_context}\n\nProvide a comprehensive analysis:"
             )
        ])

        return analysis_prompt

    def _chunk_text(self,
                    text: str,
                    chunk_size: int = 1000,
                    overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better vector storage and retrieval
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find the end of this chunk
            end = start + chunk_size

            # If not at the end of text, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + chunk_size - 200, start)
                sentence_ends = []

                for pattern in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    pos = text.rfind(pattern, search_start, end)
                    if pos != -1:
                        sentence_ends.append(pos + len(pattern))

                if sentence_ends:
                    end = max(sentence_ends)

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

            # Ensure we don't go backwards
            if start <= len(chunks[-1]) if chunks else 0:
                start = end

        return chunks

    def add_document_to_company_store(self,
                                      document_text: str,
                                      document_id: str,
                                      company_ticker: str,
                                      company_name: str = None,
                                      metadata: Dict = None):
        """
        Add a document to a company's specific vector store with chunking for large documents
        
        Args:
            document_text: Document text content
            document_id: Unique document identifier
            company_ticker: Company ticker symbol
            company_name: Company name (optional)
            metadata: Optional metadata dictionary
        """
        try:
            # Get company collection
            collection = self._get_company_collection(company_ticker,
                                                      company_name)

            # Chunk the document if it's large
            chunks = self._chunk_text(document_text,
                                      chunk_size=1500,
                                      overlap=300)

            logger.info(
                f"Chunking document {document_id} into {len(chunks)} chunks for {company_ticker}"
            )

            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create unique chunk ID
                chunk_id = f"{document_id}_chunk_{i:03d}"

                # Create embedding for this chunk
                embedding = self.embedding_model.encode(chunk)

                # Prepare metadata for this chunk
                chunk_metadata = {
                    'company_ticker': company_ticker,
                    'original_document_id': document_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'added_at': datetime.now().isoformat()
                }
                if company_name:
                    chunk_metadata['company_name'] = company_name
                if metadata:
                    chunk_metadata.update(metadata)

                # Add chunk to ChromaDB
                collection.add(documents=[chunk],
                               embeddings=[embedding.tolist()],
                               ids=[chunk_id],
                               metadatas=[chunk_metadata])

            logger.info(
                f"Added document {document_id} ({len(chunks)} chunks) to {company_ticker} collection"
            )

        except Exception as e:
            logger.error(
                f"Error adding document to {company_ticker} vector store: {e}")

    def retrieve_company_documents(self,
                                   query: str,
                                   company_ticker: str,
                                   top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant document chunks from a company's vector store
        
        Args:
            query: Search query
            company_ticker: Company ticker symbol
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks with scores, ranked by similarity
        """
        try:
            # Get company collection
            collection_key = company_ticker.upper()
            if collection_key not in self._company_collections:
                # Try to load collection
                try:
                    self._get_company_collection(company_ticker)
                except:
                    logger.warning(
                        f"No collection found for company {company_ticker}")
                    return []

            collection = self._company_collections[collection_key]

            # Check if collection has documents
            count = collection.count()
            if count == 0:
                logger.warning(
                    f"No documents found in collection for {company_ticker}")
                return []

            # Create query embedding
            query_embedding = self.embedding_model.encode(query)

            # Search in ChromaDB - retrieve more chunks initially for better ranking
            search_limit = min(top_k * 3, count)  # Get 3x more for ranking
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=search_limit)

            # Format and rank results by similarity score (lower distance = higher similarity)
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunk_data = {
                        'id':
                        results['ids'][0][i],
                        'content':
                        results['documents'][0][i],
                        'distance':
                        results['distances'][0][i],
                        'similarity_score':
                        1.0 / (1.0 + results['distances'][0][i]
                               ),  # Convert distance to similarity
                        'metadata':
                        results['metadatas'][0][i]
                        if results['metadatas'][0] else {}
                    }
                    chunks.append(chunk_data)

            # Sort by similarity score (highest first) and take top_k
            chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_chunks = chunks[:top_k]

            # Log chunk information for debugging
            logger.info(
                f"Retrieved {len(top_chunks)} top chunks for {company_ticker} from {len(chunks)} total results"
            )

            # Format final results
            documents = []
            for chunk in top_chunks:
                documents.append({
                    'id': chunk['id'],
                    'content': chunk['content'],
                    'score': chunk['similarity_score'],
                    'distance': chunk['distance'],
                    'metadata': chunk['metadata']
                })

            return documents

        except Exception as e:
            logger.error(
                f"Error retrieving documents for {company_ticker}: {e}")
            return []

    def analyze_company_ma_impact(self,
                                  query: str,
                                  company_ticker: str,
                                  top_k: int = 10) -> Dict:
        """
        Analyze M&A impact for a specific company using top-ranked chunks
        
        Args:
            query: Analysis query
            company_ticker: Company ticker symbol
            top_k: Number of top chunks to retrieve for context
            
        Returns:
            Analysis results
        """
        try:
            # Retrieve top relevant chunks
            chunks = self.retrieve_company_documents(query, company_ticker,
                                                     top_k)

            if not chunks:
                return {
                    'analysis':
                    f"No documents found for {company_ticker}. Cannot perform M&A analysis.",
                    'document_count': 0,
                    'company_ticker': company_ticker,
                    'query': query
                }

            # Sort chunks by similarity score and select top chunks for LLM context
            # Use fewer chunks but with higher relevance for LLM context
            max_context_chunks = min(
                8, len(chunks))  # Limit to avoid context overflow
            top_chunks = chunks[:max_context_chunks]

            # Calculate total context size to stay within limits
            max_context_chars = 8000  # Conservative limit for LLM context
            current_chars = 0
            selected_chunks = []

            for chunk in top_chunks:
                chunk_size = len(chunk['content'])
                if current_chars + chunk_size <= max_context_chars:
                    selected_chunks.append(chunk)
                    current_chars += chunk_size
                else:
                    # Add partial chunk if it fits
                    remaining_chars = max_context_chars - current_chars
                    if remaining_chars > 200:  # Only add if meaningful content can fit
                        partial_content = chunk[
                            'content'][:remaining_chars] + "..."
                        partial_chunk = chunk.copy()
                        partial_chunk['content'] = partial_content
                        selected_chunks.append(partial_chunk)
                    break

            # Prepare document context with chunk information
            document_context = "\n\n".join([
                f"Chunk {i+1} (Similarity: {chunk['score']:.3f}, ID: {chunk['id']}):\n{chunk['content']}"
                for i, chunk in enumerate(selected_chunks)
            ])

            # Generate analysis using LLM
            prompt = self.analysis_prompt.format_prompt(
                document_context=document_context, query=query)

            response = self.llm.invoke(prompt.to_messages())
            analysis = response.content

            return {
                'analysis':
                analysis,
                'chunk_count':
                len(selected_chunks),
                'total_chunks_found':
                len(chunks),
                'context_size_chars':
                current_chars,
                'company_ticker':
                company_ticker,
                'query':
                query,
                'retrieved_chunks': [{
                    'id': chunk['id'],
                    'similarity_score': chunk['score'],
                    'distance': chunk['distance'],
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'chunk_size': len(chunk['content'])
                } for chunk in selected_chunks]
            }

        except Exception as e:
            logger.error(
                f"Error analyzing M&A impact for {company_ticker}: {e}")
            return {
                'error': str(e),
                'company_ticker': company_ticker,
                'query': query
            }

    def get_available_companies(self) -> List[Dict]:
        """Get list of companies with data in vector store"""
        try:
            collections = self.chroma_client.list_collections()
            companies = []

            for collection in collections:
                if collection.name.startswith('company_'):
                    # Get collection details
                    coll = self.chroma_client.get_collection(collection.name)
                    metadata = collection.metadata or {}

                    companies.append({
                        'collection_name':
                        collection.name,
                        'company_ticker':
                        metadata.get('company_ticker', 'Unknown'),
                        'company_name':
                        metadata.get('company_name', 'Unknown'),
                        'document_count':
                        coll.count(),
                        'created_at':
                        metadata.get('created_at', 'Unknown')
                    })

            return sorted(companies, key=lambda x: x['company_ticker'])

        except Exception as e:
            logger.error(f"Error getting available companies: {e}")
            return []

    def search_across_companies(self,
                                query: str,
                                company_tickers: List[str] = None,
                                top_k_per_company: int = 3) -> Dict:
        """
        Search across multiple company collections with chunk-based ranking
        
        Args:
            query: Search query
            company_tickers: List of company tickers to search (None for all)
            top_k_per_company: Chunks to retrieve per company
            
        Returns:
            Aggregated search results with ranked chunks
        """
        try:
            if company_tickers:
                search_companies = company_tickers
            else:
                # Get all available companies
                available = self.get_available_companies()
                search_companies = [c['company_ticker'] for c in available]

            all_results = []
            results_by_company = {}
            companies_searched = 0
            companies_with_results = 0

            # Search each company's collection
            for company_ticker in search_companies:
                try:
                    companies_searched += 1

                    # Get top chunks for this company
                    company_chunks = self.retrieve_company_documents(
                        query=query,
                        company_ticker=company_ticker,
                        top_k=top_k_per_company *
                        2  # Get more for better selection
                    )

                    if company_chunks:
                        companies_with_results += 1

                        # Take top chunks by similarity score
                        top_chunks = company_chunks[:top_k_per_company]
                        results_by_company[company_ticker] = top_chunks

                        # Add to global results for overall ranking
                        for chunk in top_chunks:
                            chunk['company_ticker'] = company_ticker
                            all_results.append(chunk)

                except Exception as e:
                    logger.warning(f"Error searching {company_ticker}: {e}")
                    continue

            # Sort all results by similarity score across all companies
            all_results.sort(key=lambda x: x['score'], reverse=True)

            # Group top results by company for summary
            top_results_by_company = {}
            for result in all_results[:min(50, len(all_results)
                                           )]:  # Top 50 overall
                ticker = result['company_ticker']
                if ticker not in top_results_by_company:
                    top_results_by_company[ticker] = []
                top_results_by_company[ticker].append(result)

            return {
                'query': query,
                'companies_searched': companies_searched,
                'companies_with_results': companies_with_results,
                'total_documents': len(all_results),
                'results_by_company': results_by_company,
                'top_results_global':
                all_results[:20],  # Top 20 chunks globally
                'top_results_by_company': top_results_by_company
            }

        except Exception as e:
            logger.error(f"Error in cross-company search: {e}")
            return {
                'query': query,
                'error': str(e),
                'companies_searched': 0,
                'companies_with_results': 0,
                'total_documents': 0
            }

    def get_company_statistics(self, company_ticker: str) -> Dict:
        """Get statistics for a company's collection"""
        try:
            collection = self._get_company_collection(company_ticker)

            # Get basic stats
            doc_count = collection.count()
            metadata = collection.metadata

            if doc_count == 0:
                return {
                    'company_ticker': company_ticker,
                    'document_count': 0,
                    'status': 'empty'
                }

            # Get sample of documents to analyze
            sample_results = collection.query(
                query_texts=["M&A merger acquisition"],
                n_results=min(10, doc_count))

            # Analyze metadata
            filing_dates = []
            form_types = set()
            ma_scores = []

            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                for meta in sample_results['metadatas'][0]:
                    if meta:
                        if 'filing_date' in meta:
                            filing_dates.append(meta['filing_date'])
                        if 'form_type' in meta:
                            form_types.add(meta['form_type'])
                        if 'ma_score' in meta:
                            try:
                                ma_scores.append(float(meta['ma_score']))
                            except:
                                pass

            return {
                'company_ticker': company_ticker,
                'document_count': doc_count,
                'collection_metadata': metadata,
                'sample_filing_dates': filing_dates[:5],
                'form_types': list(form_types),
                'avg_ma_score':
                sum(ma_scores) / len(ma_scores) if ma_scores else None,
                'status': 'active'
            }

        except Exception as e:
            logger.error(f"Error getting statistics for {company_ticker}: {e}")
            return {
                'company_ticker': company_ticker,
                'error': str(e),
                'status': 'error'
            }

    def delete_company_collection(self, company_ticker: str) -> bool:
        """Delete a company's collection"""
        try:
            collection_name = self._normalize_company_name(company_ticker)
            self.chroma_client.delete_collection(name=collection_name)

            # Remove from cache
            collection_key = company_ticker.upper()
            if collection_key in self._company_collections:
                del self._company_collections[collection_key]

            logger.info(f"Deleted collection for {company_ticker}")
            return True

        except Exception as e:
            logger.error(
                f"Error deleting collection for {company_ticker}: {e}")
            return False

    def close(self):
        """Close connections and cleanup"""
        self._company_collections.clear()
        logger.info("Vector RAG engine closed")
