"""
Enhanced RAG GUI with Document Upload Functionality
Provides a comprehensive interface for document management and Q&A.
"""

import gradio as gr
import requests
import json
import uuid
from typing import List, Optional
import os

# Configuration
API_URL = "http://localhost:8082"

# Global user session
user_session = {"user_id": str(uuid.uuid4())}

def upload_document(file_path: str, user_id: str = None) -> dict:
    """Upload a document to the RAG system."""
    if not file_path:
        return {"error": "No file selected"}
    
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            params = {'user_id': user_id}
            response = requests.post(f"{API_URL}/documents/upload", files=files, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.text}"}
    except Exception as e:
        return {"error": f"Error uploading file: {str(e)}"}

def upload_multiple_documents(files: List[str], user_id: str = None) -> dict:
    """Upload multiple documents to the RAG system."""
    if not files:
        return {"error": "No files selected"}
    
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        files_data = []
        for file_path in files:
            if file_path and os.path.exists(file_path):
                files_data.append(('files', open(file_path, 'rb')))
        
        if not files_data:
            return {"error": "No valid files found"}
        
        params = {'user_id': user_id}
        response = requests.post(f"{API_URL}/documents/upload-multiple", files=files_data, params=params)
        
        # Close all opened files
        for _, file_obj in files_data:
            file_obj.close()
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.text}"}
    except Exception as e:
        return {"error": f"Error uploading files: {str(e)}"}

def list_user_documents(user_id: str = None) -> dict:
    """List documents in user's knowledge base."""
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        response = requests.get(f"{API_URL}/documents/list/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to list documents: {response.text}"}
    except Exception as e:
        return {"error": f"Error listing documents: {str(e)}"}

def delete_document(document_name: str, user_id: str = None) -> dict:
    """Delete a document from user's knowledge base."""
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        response = requests.delete(f"{API_URL}/documents/{user_id}/{document_name}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to delete document: {response.text}"}
    except Exception as e:
        return {"error": f"Error deleting document: {str(e)}"}

def get_vectorstore_info(user_id: str = None) -> dict:
    """Get information about user's vector store."""
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        response = requests.get(f"{API_URL}/documents/vectorstore/info/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get vectorstore info: {response.text}"}
    except Exception as e:
        return {"error": f"Error getting vectorstore info: {str(e)}"}

def clear_vectorstore(user_id: str = None) -> dict:
    """Clear all documents from user's vector store."""
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        response = requests.post(f"{API_URL}/documents/clear/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to clear vectorstore: {response.text}"}
    except Exception as e:
        return {"error": f"Error clearing vectorstore: {str(e)}"}

def ask_question(question: str, knowledge_base_type: str = "personal", user_id: str = None) -> tuple:
    """Ask a question using the RAG system."""
    if not question.strip():
        return "Please enter a question.", "", "", ""
    
    if not user_id:
        user_id = user_session["user_id"]
    
    try:
        messages = [{"role": "user", "content": question}]
        
        # Prepare request based on knowledge base type
        if knowledge_base_type == "personal":
            params = {"user_id": user_id, "use_combined": False}
        elif knowledge_base_type == "combined":
            params = {"user_id": user_id, "use_combined": True}
        else:  # default
            params = {}
        
        response = requests.post(
            f"{API_URL}/generate-answer",
            json={"messages": messages},
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract answer
            answer = ""
            for msg in result.get("messages", []):
                if msg.get("role") == "system":
                    answer = msg.get("content", "")
            
            # Format additional info
            questions = "\\n".join(result.get("questions", []))
            documents = "\\n---\\n".join(result.get("documents", []))
            kb_type = result.get("knowledge_base_type", knowledge_base_type)
            
            return answer, questions, documents, f"Knowledge Base: {kb_type}"
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            return error_msg, "", "", ""
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "", "", ""

# Gradio interface functions
def handle_file_upload(file):
    """Handle single file upload."""
    if file is None:
        return "No file selected", ""
    
    result = upload_document(file.name)
    
    if "error" in result:
        return f"❌ {result['error']}", ""
    else:
        success_msg = f"✅ {result['message']}\\n"
        success_msg += f"📄 Document: {result['document_name']}\\n"
        success_msg += f"📊 Chunks: {result['chunks_created']}\\n"
        success_msg += f"👤 User ID: {result['user_id']}"
        
        # Update global user session
        user_session["user_id"] = result["user_id"]
        
        return success_msg, get_document_list_display()

def handle_multiple_file_upload(files):
    """Handle multiple file upload."""
    if not files:
        return "No files selected", ""
    
    file_paths = [f.name for f in files if f is not None]
    result = upload_multiple_documents(file_paths)
    
    if "error" in result:
        return f"❌ {result['error']}", ""
    else:
        success_msg = f"✅ Processed {result['total_files']} files\\n"
        success_msg += f"📈 Successful: {result['successful_uploads']}\\n"
        success_msg += f"❌ Failed: {result['failed_uploads']}\\n"
        success_msg += f"👤 User ID: {result['user_id']}"
        
        # Update global user session
        user_session["user_id"] = result["user_id"]
        
        return success_msg, get_document_list_display()

def get_document_list_display():
    """Get formatted display of user documents."""
    result = list_user_documents()
    
    if "error" in result:
        return f"❌ {result['error']}"
    
    if not result.get("documents"):
        return "📭 No documents uploaded yet."
    
    display = f"📚 **Knowledge Base ({result['total_documents']} documents)**\\n\\n"
    
    for doc in result["documents"]:
        display += f"📄 **{doc['name']}**\\n"
        display += f"   📊 Chunks: {doc['chunks']}\\n"
        display += f"   🔧 Type: {doc['document_type']}\\n\\n"
    
    return display

def handle_document_deletion(document_name):
    """Handle document deletion."""
    if not document_name.strip():
        return "Please enter a document name", ""
    
    result = delete_document(document_name.strip())
    
    if "error" in result:
        return f"❌ {result['error']}", get_document_list_display()
    else:
        success_msg = f"✅ {result['message']}\\n"
        success_msg += f"🗑️ Deleted {result['deleted_chunks']} chunks"
        return success_msg, get_document_list_display()

def handle_vectorstore_clear():
    """Handle clearing the vectorstore."""
    result = clear_vectorstore()
    
    if "error" in result:
        return f"❌ {result['error']}", ""
    else:
        return f"✅ {result['message']}", "📭 No documents uploaded yet."

def get_vectorstore_info_display():
    """Get vectorstore information display."""
    result = get_vectorstore_info()
    
    if "error" in result:
        return f"❌ {result['error']}"
    
    info = f"📊 **Vector Store Information**\\n\\n"
    info += f"👤 User ID: {result['user_id']}\\n"
    info += f"📚 Total Documents: {result['total_documents']}\\n"
    info += f"📊 Total Chunks: {result['total_chunks']}\\n"
    info += f"📋 Status: {result['status']}\\n"
    
    return info

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="RAG Knowledge Base Manager", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧠 RAG Knowledge Base Manager")
        gr.Markdown("Upload documents and ask questions using Retrieval-Augmented Generation (RAG)")
        
        with gr.Tab("📤 Document Upload"):
            gr.Markdown("## Upload Documents to Your Knowledge Base")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Single File Upload")
                    file_upload = gr.File(
                        label="Select Document (PDF, DOCX, TXT, MD)",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    upload_btn = gr.Button("📤 Upload Document", variant="primary")
                    upload_result = gr.Textbox(label="Upload Result", lines=4)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Multiple Files Upload")
                    files_upload = gr.File(
                        label="Select Multiple Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    multi_upload_btn = gr.Button("📤 Upload Multiple", variant="primary")
                    multi_upload_result = gr.Textbox(label="Batch Upload Result", lines=4)
        
        with gr.Tab("📚 Knowledge Base"):
            gr.Markdown("## Manage Your Knowledge Base")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Document List")
                    doc_list = gr.Markdown(value=get_document_list_display())
                    refresh_btn = gr.Button("🔄 Refresh List")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🗑️ Delete Document")
                    doc_to_delete = gr.Textbox(
                        label="Document Name to Delete",
                        placeholder="Enter exact document name..."
                    )
                    delete_btn = gr.Button("🗑️ Delete Document", variant="secondary")
                    delete_result = gr.Textbox(label="Delete Result", lines=2)
                    
                    gr.Markdown("### 📊 Vector Store Info")
                    info_btn = gr.Button("📊 Get Info")
                    vectorstore_info = gr.Markdown()
                    
                    gr.Markdown("### 🧹 Clear All")
                    clear_btn = gr.Button("🧹 Clear All Documents", variant="stop")
                    clear_result = gr.Textbox(label="Clear Result", lines=2)
        
        with gr.Tab("❓ Ask Questions"):
            gr.Markdown("## Ask Questions About Your Documents")
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3
                    )
                    
                    knowledge_base_choice = gr.Radio(
                        choices=["personal", "combined", "default"],
                        label="Knowledge Base",
                        value="personal",
                        info="Personal: Your documents only | Combined: Your + default docs | Default: Built-in docs only"
                    )
                    
                    ask_btn = gr.Button("🤖 Ask Question", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=8,
                        max_lines=20
                    )
                    
                    with gr.Accordion("🔍 Query Details", open=False):
                        questions_output = gr.Textbox(label="Generated Queries", lines=3)
                        documents_output = gr.Textbox(label="Source Documents", lines=4)
                        kb_info_output = gr.Textbox(label="Knowledge Base Info", lines=1)
        
        # Event handlers
        upload_btn.click(
            handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_result, doc_list]
        )
        
        multi_upload_btn.click(
            handle_multiple_file_upload,
            inputs=[files_upload],
            outputs=[multi_upload_result, doc_list]
        )
        
        refresh_btn.click(
            lambda: get_document_list_display(),
            outputs=[doc_list]
        )
        
        delete_btn.click(
            handle_document_deletion,
            inputs=[doc_to_delete],
            outputs=[delete_result, doc_list]
        )
        
        info_btn.click(
            get_vectorstore_info_display,
            outputs=[vectorstore_info]
        )
        
        clear_btn.click(
            handle_vectorstore_clear,
            outputs=[clear_result, doc_list]
        )
        
        ask_btn.click(
            ask_question,
            inputs=[question_input, knowledge_base_choice],
            outputs=[answer_output, questions_output, documents_output, kb_info_output]
        )
        
        # Auto-refresh document list on tab switch
        app.load(
            lambda: get_document_list_display(),
            outputs=[doc_list]
        )
    
    return app

if __name__ == "__main__":
    # Display user session info
    print(f"🚀 Starting RAG Knowledge Base Manager")
    print(f"👤 User Session ID: {user_session['user_id']}")
    print(f"🔗 API URL: {API_URL}")
    
    # Create and launch the interface
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )