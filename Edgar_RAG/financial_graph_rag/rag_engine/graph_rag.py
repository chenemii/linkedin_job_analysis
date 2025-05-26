"""
Graph RAG Engine

Main engine that combines knowledge graph querying with retrieval-augmented generation
for analyzing M&A impacts on organizational structure.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..config import settings
from ..knowledge_graph.graph_db import GraphDatabaseManager
from ..knowledge_graph.entities import EntityType, RelationshipType

logger = logging.getLogger(__name__)


class GraphRAGEngine:
    """
    Graph-enhanced Retrieval Augmented Generation engine for financial analysis
    """

    def __init__(self):
        """Initialize the Graph RAG engine"""
        # Initialize components
        self.graph_db = GraphDatabaseManager()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory)
        self.collection = self._get_or_create_collection()

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Create prompts
        self.analysis_prompt = self._create_analysis_prompt()
        self.graph_query_prompt = self._create_graph_query_prompt()

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

    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            collection = self.chroma_client.get_collection(
                name=settings.chroma_collection_name)
        except:
            collection = self.chroma_client.create_collection(
                name=settings.chroma_collection_name,
                metadata={
                    "description": "Financial documents for M&A analysis"
                })

        return collection

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for M&A impact analysis"""
        template = """
        You are a financial analyst expert specializing in mergers & acquisitions and organizational structure analysis.
        
        Using the provided context from knowledge graph and document retrieval, analyze the impact of M&A activities 
        on organizational structure for the specified company.
        
        KNOWLEDGE GRAPH CONTEXT:
        {graph_context}
        
        DOCUMENT CONTEXT:
        {document_context}
        
        QUERY: {query}
        
        Provide a comprehensive analysis covering:
        
        1. **M&A Activity Summary**
           - Key transactions identified
           - Timeline and transaction values
           - Strategic rationale mentioned
        
        2. **Organizational Structure Impact**
           - Changes in corporate hierarchy
           - Subsidiary integration or restructuring
           - Division/business unit changes
           - Personnel and leadership changes
        
        3. **Integration Challenges & Synergies**
           - Integration challenges mentioned
           - Expected or realized synergies
           - Cultural integration aspects
           - Operational integration details
        
        4. **Strategic Implications**
           - Strategic positioning changes
           - Market expansion or consolidation
           - Competitive advantages gained
           - Risk factors identified
        
        5. **Key Insights & Recommendations**
           - Critical success factors
           - Areas requiring attention
           - Recommendations for stakeholders
        
        Format your response in clear sections with specific evidence from the provided context.
        Include relevant dates, transaction values, and specific organizational changes mentioned.
        """

        return ChatPromptTemplate.from_template(template)

    def _create_graph_query_prompt(self) -> ChatPromptTemplate:
        """Create prompt for generating graph queries"""
        template = """
        You are an expert at translating financial analysis questions into Neo4j Cypher queries.
        
        Given this question about M&A and organizational structure analysis:
        
        QUESTION: {question}
        
        IMPORTANT - CURRENT DATABASE SCHEMA:
        The knowledge graph currently contains:
        - Node labels: Entity, Company (with properties: id, name, ticker, cik, sector, industry, headquarters)
        - Relationship types: RELATED (general relationship with 'type' property)
        
        IMPORTANT CONSTRAINTS:
        - NO M&A transaction data exists yet (no Transaction nodes, Subsidiary nodes, or M&A relationships)
        - NO organizational structure data exists yet (no Division nodes, Person nodes)
        - Only basic S&P 500 company information is available
        
        INSTRUCTIONS:
        1. If the question asks for M&A data that doesn't exist, return a query that finds relevant companies instead
        2. Use only existing node labels: Company, Entity
        3. Use only existing relationship type: RELATED  
        4. Focus on company sector, industry, or basic company information
        5. Return a helpful query that works with current data
        
        Examples of good queries for current schema:
        - Find companies in specific sectors
        - Find companies by name or ticker
        - Compare companies within the same industry
        
        Generate a Cypher query that would help answer this question with available data.
        Return ONLY the Cypher query, no explanation or code blocks.
        """

        return ChatPromptTemplate.from_template(template)

    def add_document_to_vector_store(self,
                                     document_text: str,
                                     document_id: str,
                                     metadata: Dict = None):
        """
        Add a document to the vector store
        
        Args:
            document_text: Document text content
            document_id: Unique document identifier
            metadata: Optional metadata dictionary
        """
        try:
            # Create embedding
            embedding = self.embedding_model.encode(document_text)

            # Add to ChromaDB
            self.collection.add(documents=[document_text],
                                embeddings=[embedding.tolist()],
                                ids=[document_id],
                                metadatas=[metadata or {}])

            logger.info(f"Added document {document_id} to vector store")

        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")

    def retrieve_relevant_documents(self,
                                    query: str,
                                    top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from vector store
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k)

            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'score': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                    if results['metadatas'][0] else {}
                })

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def query_knowledge_graph(self,
                              question: str,
                              company_id: Optional[str] = None) -> Dict:
        """
        Query the knowledge graph for relevant information
        
        Args:
            question: Analysis question
            company_id: Optional company identifier
            
        Returns:
            Dictionary with graph query results
        """
        try:
            # Generate Cypher query using LLM
            query_prompt = self.graph_query_prompt.format(question=question)

            # Log the prompt being sent to the LLM
            logger.info("=" * 50)
            logger.info("PROMPT SENT TO OLLAMA MODEL:")
            logger.info("=" * 50)
            logger.info(query_prompt)
            logger.info("=" * 50)

            response = self.llm.invoke(query_prompt)
            raw_cypher_query = response.content.strip()

            # Log the raw response from LLM
            logger.info("RAW LLM RESPONSE:")
            logger.info("=" * 30)
            logger.info(raw_cypher_query)
            logger.info("=" * 30)

            # Clean the Cypher query - remove markdown formatting if present
            cypher_query = self._clean_cypher_query(raw_cypher_query)

            # Log the cleaned query
            logger.info("CLEANED CYPHER QUERY:")
            logger.info("=" * 30)
            logger.info(cypher_query)
            logger.info("=" * 30)

            # Execute the query
            results = self.graph_db.execute_custom_query(cypher_query)

            # If company_id provided, also get specific company context
            if company_id:
                # Get company structure
                company_structure = self.graph_db.find_company_structure(
                    company_id)

                # Get M&A transactions
                ma_transactions = self.graph_db.find_ma_transactions(
                    company_id)

                return {
                    'generated_query': cypher_query,
                    'query_results': results,
                    'company_structure': company_structure,
                    'ma_transactions': ma_transactions
                }

            return {'generated_query': cypher_query, 'query_results': results}

        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return {'error': str(e)}

    def _clean_cypher_query(self, raw_query: str) -> str:
        """
        Clean Cypher query by removing markdown formatting and other artifacts
        
        Args:
            raw_query: Raw query string from LLM
            
        Returns:
            Clean Cypher query string
        """
        # Remove markdown code block formatting
        if raw_query.startswith('```cypher'):
            # Remove opening ```cypher
            raw_query = raw_query[9:].strip()
        elif raw_query.startswith('```'):
            # Remove opening ```
            raw_query = raw_query[3:].strip()

        # Remove closing ```
        if raw_query.endswith('```'):
            raw_query = raw_query[:-3].strip()

        # Remove any leading/trailing whitespace
        raw_query = raw_query.strip()

        return raw_query

    def analyze_ma_impact(self,
                          query: str,
                          company_ticker: Optional[str] = None,
                          company_cik: Optional[str] = None,
                          years: Optional[List[int]] = None) -> Dict:
        """
        Comprehensive M&A impact analysis using Graph RAG
        
        Args:
            query: Analysis question or topic
            company_ticker: Optional company ticker symbol
            company_cik: Optional company CIK
            years: Optional list of years to focus on
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting M&A impact analysis for query: {query}")

        # Step 1: Retrieve relevant documents from vector store
        relevant_docs = self.retrieve_relevant_documents(query, top_k=5)

        # Step 2: Resolve company ID - prefer CIK, but if only ticker provided, look up the CIK
        company_id = company_cik
        if not company_id and company_ticker:
            # Look up the company's CIK from ticker
            lookup_result = self.graph_db.execute_custom_query(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.id as id",
                {"ticker": company_ticker})
            if lookup_result:
                company_id = lookup_result[0]['id']
                logger.info(
                    f"Resolved ticker {company_ticker} to company ID: {company_id}"
                )
            else:
                company_id = company_ticker  # fallback to ticker

        # Step 3: Query knowledge graph for structured information
        graph_context = self.query_knowledge_graph(query, company_id)

        # Step 4: Get additional context if years specified
        temporal_context = {}
        if company_id and years:
            temporal_context = self.graph_db.analyze_ma_impact_on_structure(
                company_id, years)

        # Step 5: Prepare context for LLM analysis
        document_context = "\n\n".join([
            f"Document {doc['id']} (Score: {doc['score']:.3f}):\n{doc['content'][:1000]}..."
            for doc in relevant_docs
        ])

        graph_context_str = json.dumps(graph_context, indent=2)

        # Step 6: Generate comprehensive analysis
        analysis_prompt = self.analysis_prompt.format(
            graph_context=graph_context_str,
            document_context=document_context,
            query=query)

        # Log the analysis prompt being sent to the LLM
        logger.info("=" * 60)
        logger.info("ANALYSIS PROMPT SENT TO OLLAMA MODEL:")
        logger.info("=" * 60)
        logger.info(analysis_prompt)
        logger.info("=" * 60)

        analysis_response = self.llm.invoke(analysis_prompt)

        # Log the analysis response
        logger.info("ANALYSIS LLM RESPONSE:")
        logger.info("=" * 40)
        logger.info(analysis_response.content)
        logger.info("=" * 40)

        # Step 7: Compile results
        results = {
            'query': query,
            'analysis': analysis_response.content,
            'supporting_evidence': {
                'relevant_documents': relevant_docs,
                'graph_context': graph_context,
                'temporal_context': temporal_context
            },
            'metadata': {
                'company_id': company_id,
                'company_ticker': company_ticker,
                'company_cik': company_cik,
                'years_analyzed': years,
                'documents_retrieved': len(relevant_docs),
                'graph_query_generated':
                graph_context.get('generated_query', '')
            }
        }

        logger.info("Completed M&A impact analysis")
        return results

    def get_company_ma_timeline(self,
                                company_id: str,
                                start_year: Optional[int] = None,
                                end_year: Optional[int] = None) -> Dict:
        """
        Get M&A timeline for a specific company
        
        Args:
            company_id: Company identifier (CIK or ticker)
            start_year: Optional start year filter
            end_year: Optional end year filter
            
        Returns:
            Timeline of M&A activities with organizational impact
        """
        # Get M&A transactions
        transactions = self.graph_db.find_ma_transactions(company_id)

        # Filter by years if specified
        if start_year or end_year:
            filtered_transactions = []
            for trans in transactions:
                trans_year = None
                if trans.get('announcement_date'):
                    try:
                        trans_year = int(trans['announcement_date'][:4])
                    except:
                        continue

                if start_year and trans_year and trans_year < start_year:
                    continue
                if end_year and trans_year and trans_year > end_year:
                    continue

                filtered_transactions.append(trans)

            transactions = filtered_transactions

        # Get company structure evolution if we have data
        company_structure = self.graph_db.find_company_structure(company_id)

        return {
            'company_id': company_id,
            'transactions': transactions,
            'current_structure': company_structure,
            'timeline_summary': {
                'total_transactions':
                len(transactions),
                'years_covered':
                list(
                    set([
                        trans['announcement_date'][:4]
                        for trans in transactions
                        if trans.get('announcement_date')
                    ]))
            }
        }

    def search_companies_by_ma_activity(
            self,
            activity_type: str = "acquisition",
            min_value: Optional[float] = None,
            sector: Optional[str] = None) -> List[Dict]:
        """
        Search for companies based on M&A activity criteria
        
        Args:
            activity_type: Type of M&A activity to search for
            min_value: Minimum transaction value
            sector: Company sector filter
            
        Returns:
            List of companies matching criteria
        """
        # Build Cypher query based on criteria
        query_parts = [
            "MATCH (c:Company)-[r:RELATED]->(t:Transaction)",
            f"WHERE r.type CONTAINS '{activity_type}'"
        ]

        if min_value:
            query_parts.append(f"AND t.transaction_value >= {min_value}")

        if sector:
            query_parts.append(f"AND c.sector = '{sector}'")

        query_parts.extend(
            ["RETURN c, collect(t) as transactions", "ORDER BY c.name"])

        cypher_query = "\n".join(query_parts)

        # Execute query
        results = self.graph_db.execute_custom_query(cypher_query)

        return results

    def close(self):
        """Close database connections"""
        if self.graph_db:
            self.graph_db.close()
