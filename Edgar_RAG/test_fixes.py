#!/usr/bin/env python3
"""
Test script to verify performance fixes and error handling
"""

import os
import time
import logging

# Set tokenizers parallelism before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_initialization():
    """Test system initialization with fixes"""
    print("🧪 Testing system initialization...")
    
    try:
        start_time = time.time()
        
        # Import and initialize system
        from financial_graph_rag.core import FinancialVectorRAG
        system = FinancialVectorRAG()
        
        init_time = time.time() - start_time
        print(f"✅ System initialized successfully in {init_time:.2f}s")
        
        return system
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return None

def test_skill_analysis(system):
    """Test skill shortage analysis with retry logic"""
    print("\n🧪 Testing skill shortage analysis...")
    
    try:
        start_time = time.time()
        
        # Test skill shortage analysis
        results = system.analyze_company_skill_shortage("ABT")
        
        analysis_time = time.time() - start_time
        print(f"✅ Skill shortage analysis completed in {analysis_time:.2f}s")
        
        if 'error' in results:
            print(f"⚠️  Analysis returned error: {results['error']}")
        else:
            print(f"✅ Analysis successful - {results.get('chunk_count', 0)} chunks analyzed")
        
        return results
        
    except Exception as e:
        print(f"❌ Skill shortage analysis failed: {e}")
        return None

def test_vector_operations(system):
    """Test vector store operations"""
    print("\n🧪 Testing vector store operations...")
    
    try:
        # Test document retrieval
        docs = system.rag_engine.retrieve_company_documents(
            query="skill shortage talent gap",
            company_ticker="ABT",
            top_k=3
        )
        
        print(f"✅ Retrieved {len(docs)} documents from vector store")
        
        # Test company list
        companies = system.rag_engine.get_available_companies()
        print(f"✅ Found {len(companies)} companies in vector store")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store operations failed: {e}")
        return False

def test_llm_connection(system):
    """Test LLM connection with retry logic"""
    print("\n🧪 Testing LLM connection...")
    
    try:
        # Test simple LLM call
        from langchain.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Say 'Hello, world!'")
        ])
        
        messages = prompt.format_messages()
        response = system.rag_engine.llm.invoke(messages)
        
        print(f"✅ LLM connection successful: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting performance fix verification tests...")
    print("=" * 60)
    
    # Test 1: System initialization
    system = test_system_initialization()
    if not system:
        print("❌ Cannot continue without system initialization")
        return
    
    # Test 2: Vector store operations
    vector_success = test_vector_operations(system)
    
    # Test 3: LLM connection
    llm_success = test_llm_connection(system)
    
    # Test 4: Skill shortage analysis (only if LLM works)
    if llm_success:
        analysis_results = test_skill_analysis(system)
    else:
        print("\n⚠️  Skipping skill shortage analysis due to LLM connection issues")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"✅ System Initialization: {'PASS' if system else 'FAIL'}")
    print(f"✅ Vector Store Operations: {'PASS' if vector_success else 'FAIL'}")
    print(f"✅ LLM Connection: {'PASS' if llm_success else 'FAIL'}")
    
    if llm_success:
        print(f"✅ Skill Shortage Analysis: {'PASS' if analysis_results and 'error' not in analysis_results else 'FAIL'}")
    
    print("\n🎉 Performance fixes verification completed!")

if __name__ == "__main__":
    main() 