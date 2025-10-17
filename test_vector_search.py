"""
Simple Vector Search Test Script
Test basic functionality without complex dependencies.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ragchallenge.api.config import settings

def test_vector_search():
    """Test basic vector search functionality."""
    print("🔍 Testing Vector Search...")
    
    try:
        # Initialize embeddings
        print("📊 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.embedding_model_device}
        )
        print("✅ Embeddings loaded successfully")
        
        # Load existing vector store
        print("📚 Loading vector store...")
        vectorstore = Chroma(
            persist_directory="data/vectorstore",
            embedding_function=embeddings
        )
        
        # Get total documents
        collection = vectorstore._collection
        results = collection.get()
        print(f"📄 Found {len(results['ids'])} documents in vector store")
        
        if len(results['ids']) == 0:
            print("⚠️  Vector store is empty!")
            return False
        
        # Test search
        print("🔍 Testing similarity search...")
        test_query = "How to initialize git repository?"
        search_results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"📋 Search Results for: '{test_query}'")
        for i, doc in enumerate(search_results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
            print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in vector search test: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_search()
    if success:
        print("\n✅ Vector search is working correctly!")
    else:
        print("\n❌ Vector search test failed!")