"""
Complete RAG Pipeline Test
Test the full retrieval-augmented generation pipeline.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from ragchallenge.api.config import settings

def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("🧠 Testing Complete RAG Pipeline...")
    
    try:
        # 1. Initialize embeddings
        print("📊 Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.embedding_model_device}
        )
        
        # 2. Load vector store
        print("📚 Loading vector store...")
        vectorstore = Chroma(
            persist_directory="data/vectorstore",
            embedding_function=embeddings
        )
        
        # 3. Initialize Gemini LLM
        print("🤖 Initializing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            max_output_tokens=2048,
        )
        
        # 4. Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical assistant specializing in tools like Git, Conda, and regular expressions (Regex).
Your task is to provide accurate, concise, and context-specific answers to technical questions.

Using the context provided below, answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        # 5. Test query
        test_question = "How do I initialize a new Git repository?"
        print(f"❓ Question: {test_question}")
        
        # 6. Retrieve relevant documents
        print("🔍 Retrieving relevant documents...")
        search_results = vectorstore.similarity_search(test_question, k=3)
        context = "\n\n".join([doc.page_content for doc in search_results])
        
        print(f"📄 Retrieved {len(search_results)} documents")
        print(f"📝 Context preview: {context[:200]}...")
        
        # 7. Generate answer using LLM
        print("🤖 Generating answer with Gemini...")
        
        # Create the chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": test_question
        })
        
        print(f"\n🎯 Generated Answer:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        # 8. Display source information
        print(f"\n📚 Sources:")
        for i, doc in enumerate(search_results, 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown')} (chunk {doc.metadata.get('chunk', 'N/A')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\n🎉 RAG pipeline is working correctly!")
    else:
        print("\n❌ RAG pipeline test failed!")