"""
Enhanced RAG Chat Application

A Streamlit application for document processing and AI-powered question answering
using Retrieval-Augmented Generation (RAG) with Ollama.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Import our utility modules
from utils.rag import RAGSystem
from utils.chat_interface import ChatInterface

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    
    # Initialize chat interface
    chat_interface = ChatInterface()
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 RAG Chat Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents and ask questions with AI-powered answers</p>', unsafe_allow_html=True)
    
    # Initialize RAG system if not exists
    if 'rag_system' not in st.session_state or st.session_state.rag_system is None:
        try:
            st.session_state.rag_system = RAGSystem()
            chat_interface.display_info_message("RAG system initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            st.stop()
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader with a consistent key
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files",
            key="file_uploader"
        )
        
        # Initialize last_uploaded_files if it doesn't exist
        if 'last_uploaded_files' not in st.session_state:
            st.session_state.last_uploaded_files = []
        
        # Process button
        process_clicked = st.button("🔄 Process Documents(Manually)", help="Click to process uploaded files")
        
        # Process files if we have new uploads or the process button was clicked
        if uploaded_files:
            current_files = [f.name for f in uploaded_files]
            
            # Check if files are new or process button was clicked
            if (current_files != st.session_state.last_uploaded_files) or process_clicked:
                st.session_state.last_uploaded_files = current_files
                process_uploaded_files(uploaded_files, chat_interface)
        elif process_clicked and not uploaded_files:
            chat_interface.display_warning_message("No files selected for processing.")
        
        # Clear button to reset file uploader
        if st.button("❌ Clear Uploaded Files"):
            st.session_state.last_uploaded_files = []
            st.rerun()
        
        # Build index button (manual fallback)
        if st.session_state.documents_processed:
            successful_docs = [doc for doc in st.session_state.documents_processed if doc['success']]
            if successful_docs and not st.session_state.rag_system.is_indexed:
                if st.button("🔍 Build Search Index (Manual)", type="primary", key="build_search_index_btn"):
                    build_search_index(chat_interface)
        
        # Clear all button
        if st.button("🗑️ Clear All", key="clear_all_btn"):
            clear_all_data(chat_interface)
        
        # Debug button
        if st.button("🔍 Debug System State", key="debug_state_btn"):
            debug_system_state(chat_interface)
        
        # System status
        st.header("📊 System Status")
        chat_interface.display_system_status(st.session_state.rag_system)
        
        # Document info
        if st.session_state.documents_processed:
            chat_interface.display_document_info()
    
    # Main chat area
    st.header("💬 Chat Interface")
    
    # Display current status
    if st.session_state.documents_processed:
        successful_docs = [doc for doc in st.session_state.documents_processed if doc.get('success', False)]
        if successful_docs:
            total_chunks = sum(doc.get('chunks_created', 0) for doc in successful_docs)
            # Remove Streamlit banners, rely on notifications only
            # if st.session_state.rag_system.is_indexed:
            #     st.success(f"✅ Ready! {len(successful_docs)} documents processed ({total_chunks} chunks) and indexed. You can ask questions!")
            # else:
            #     st.warning(f"⚠️ Documents processed but index not built. {len(successful_docs)} documents ({total_chunks} chunks) ready for indexing.")
            if not st.session_state.rag_system.is_indexed:
                if st.button("🔍 Build Search Index Now", type="primary"):
                    build_search_index(chat_interface)
        # else:
        #     st.error("❌ Document processing failed. Please check your files and try again.")
    # else:
    #     st.info("📁 No documents uploaded yet. Use the sidebar to upload and process documents.")
    
    # Display chat history
    chat_interface.display_chat_history()
    
    # Chat controls
    chat_interface.display_chat_controls()
    
    # User input
    if user_input := chat_interface.get_user_input():
        st.session_state.show_notification = None  # Clear any old notification before processing
        # Add user message to chat
        chat_interface.add_user_message(user_input)
        # Process the question
        process_user_question(user_input, chat_interface)

    if st.session_state.get("show_notification"):
        st.session_state.show_notification = None

def process_uploaded_files(uploaded_files, chat_interface):
    """Process uploaded files and add them to the RAG system."""
    if not uploaded_files:
        chat_interface.display_error_message("No files selected for upload.")
        return
    
    print(f"DEBUG: Starting to process {len(uploaded_files)} files")
    
    # Clear previous documents
    st.session_state.rag_system.clear_documents()
    st.session_state.documents_processed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            print(f"DEBUG: Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            print(f"DEBUG: Saved file to {tmp_file_path}")
            
            try:
                # Process the document
                result = st.session_state.rag_system.process_document(tmp_file_path)
                print(f"DEBUG: Document processing result: {result}")
                
                # Store result
                result['file_name'] = uploaded_file.name
                st.session_state.documents_processed.append(result)
                
                if result.get('success', False):
                    chat_interface.display_success_message(
                        f"Successfully processed {uploaded_file.name} ({result.get('chunks_created', 0)} chunks)"
                    )
                    print(f"DEBUG: Successfully processed {uploaded_file.name}")
                else:
                    error_msg = f"Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}"
                    chat_interface.display_error_message(error_msg)
                    print(f"DEBUG: {error_msg}")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {tmp_file_path}: {e}")
            
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
        except Exception as e:
            error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
            print(f"DEBUG: Exception during processing: {error_msg}")
            chat_interface.display_error_message(error_msg)
            st.session_state.documents_processed.append({
                'success': False,
                'error': str(e),
                'file_name': uploaded_file.name,
                'chunks_created': 0,
                'word_count': 0,
                'metadata': {}
            })
    
    # Show summary after processing all files
    successful_docs = [doc for doc in st.session_state.documents_processed if doc.get('success', False)]
    print(f"DEBUG: Processing complete. {len(successful_docs)} successful out of {len(st.session_state.documents_processed)} total")
    
    if successful_docs:
        total_chunks = sum(doc.get('chunks_created', 0) for doc in successful_docs)
        success_msg = f"Processing complete! {len(successful_docs)} documents processed with {total_chunks} total chunks."
        chat_interface.display_success_message(success_msg)
        print(f"DEBUG: {success_msg}")
        
        # Automatically build the search index after successful processing
        print("DEBUG: Starting automatic index building...")
        chat_interface.display_info_message("Building search index automatically...")
        build_search_index(chat_interface)
    else:
        error_msg = "No documents were successfully processed."
        chat_interface.display_error_message(error_msg)
        print(f"DEBUG: {error_msg}")
    
    status_text.empty()
    progress_bar.empty()


def build_search_index(chat_interface):
    """Build the search index from processed documents."""
    try:
        with st.spinner("Building search index..."):
            result = st.session_state.rag_system.build_index()
            if result['success']:
                chat_interface.display_success_message("Search index built successfully!")
            else:
                chat_interface.display_error_message(f"Failed to build search index: {result.get('error', 'Unknown error')}")
    except Exception as e:
        chat_interface.display_error_message(f"Error building search index: {str(e)}")


def clear_all_data(chat_interface):
    """Clear all data and reset the application state."""
    if 'rag_system' in st.session_state:
        st.session_state.rag_system.clear_documents()
    st.session_state.documents_processed = []
    st.session_state.messages = []
    chat_interface.display_success_message("All data has been cleared.")


def debug_system_state(chat_interface):
    """Display debug information about the system state."""
    debug_info = {
        'rag_system_initialized': 'rag_system' in st.session_state and st.session_state.rag_system is not None,
        'documents_processed': len(st.session_state.get('documents_processed', [])),
        'messages_count': len(st.session_state.get('messages', [])),
        'is_indexed': st.session_state.get('rag_system', {}).is_indexed if 'rag_system' in st.session_state else False
    }
    
    debug_text = "## 🐛 Debug Information\n\n"
    for key, value in debug_info.items():
        debug_text += f"- **{key}**: `{value}`\n"
    
    chat_interface.add_system_message(debug_text, "info")


def process_user_question(question, chat_interface):
    """Process a user question and generate a response using the RAG system."""
    if not st.session_state.rag_system.is_indexed:
        chat_interface.display_error_message("Please build the search index before asking questions.")
        return
    
    try:
        with st.spinner("Generating response..."):
            response = st.session_state.rag_system.ask_question(question)
            if response.get('success', False):
                # Use just the answer without any additional formatting
                formatted_response = response['answer']
                
                # Add the formatted message
                chat_interface.add_assistant_message(
                    formatted_response,
                    metadata={
                        'context_docs': response.get('context_docs', 0),
                        'llm_success': response.get('llm_success', False)
                    },
                    is_html=True
                )
                
                # Display relevant documents if available
                if response.get('relevant_documents'):
                    with st.expander(f"📄 View relevant document excerpts ({len(response['relevant_documents'])})"):
                        for i, doc in enumerate(response['relevant_documents'], 1):
                            st.markdown(f"### 📄 Excerpt {i} (Relevance: {doc.get('score', 0):.1%})")
                            st.markdown("---")
                            st.markdown(f"{doc.get('content', 'No content')}")
                            st.markdown("\n")
            else:
                error_msg = response.get('error', 'Failed to generate response')
                chat_interface.display_error_message(f"❌ {error_msg}")
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"DEBUG: {error_msg}")
        chat_interface.display_error_message(error_msg)


def debug_system_state(chat_interface: ChatInterface):
    """Debug the system state and verify everything is working correctly."""
    
    debug_info = []
    debug_info.append("🔍 **System Debug Information:**")
    
    # Check RAG system
    if st.session_state.rag_system:
        debug_info.append("✅ RAG system initialized")
        status = st.session_state.rag_system.get_system_status()
        debug_info.append(f"   - Documents loaded: {status['documents_loaded']}")
        debug_info.append(f"   - Indexed: {status['is_indexed']}")
        debug_info.append(f"   - LLM connected: {status['llm_connected']}")
        debug_info.append(f"   - Model: {status['model_name']}")
    else:
        debug_info.append("❌ RAG system not initialized")
    
    # Check documents processed
    if st.session_state.documents_processed:
        debug_info.append(f"✅ Documents processed: {len(st.session_state.documents_processed)}")
        successful_docs = [doc for doc in st.session_state.documents_processed if doc.get('success', False)]
        debug_info.append(f"   - Successful: {len(successful_docs)}")
        debug_info.append(f"   - Failed: {len(st.session_state.documents_processed) - len(successful_docs)}")
        
        for i, doc in enumerate(st.session_state.documents_processed):
            status = "✅" if doc.get('success', False) else "❌"
            debug_info.append(f"   {status} {doc.get('file_name', 'Unknown')}: {doc.get('chunks_created', 0)} chunks")
        else:
            debug_info.append("❌ No documents processed")
    
    # Check chat messages
    if st.session_state.messages:
        debug_info.append(f"✅ Chat messages: {len(st.session_state.messages)}")
    else:
        debug_info.append("❌ No chat messages")
    
    # Display debug information
    debug_text = "\n".join(debug_info)
    chat_interface.add_system_message(debug_text, "debug")
    
    # Also display in the main area
    st.info(debug_text)

if __name__ == "__main__":
    main()
