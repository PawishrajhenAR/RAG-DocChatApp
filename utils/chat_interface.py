"""
Enhanced Chat Interface

This module provides a clean chat interface for the RAG system.
It handles conversation management, message formatting, and user interactions.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime

class ChatInterface:
    """
    Enhanced chat interface for RAG system interactions.
    
    Features:
    - Conversation history management
    - Message formatting and display
    - User input validation
    - Error handling and user feedback
    """
    
    def __init__(self):
        """Initialize the chat interface."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state for chat."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = []
    
    def display_chat_history(self):
        """Display the chat message history."""
        print(f"DEBUG: Displaying chat history. Total messages: {len(st.session_state.messages)}")
        
        for i, message in enumerate(st.session_state.messages):
            print(f"DEBUG: Displaying message {i+1}: {message.get('role', 'unknown')} - {message.get('content', '')[:50]}...")
            with st.chat_message(message["role"]):
                if message.get('is_html'):
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
    
    def add_user_message(self, content: str):
        """Add a user message to the chat history."""
        if content.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None, is_html: bool = False):
        """
        Add an assistant message to the chat history.
        
        Args:
            content: The message content to display
            metadata: Optional metadata to include with the message
            is_html: If True, the content will be treated as HTML and rendered with st.markdown
        """
        if content.strip():
            message = {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "is_html": is_html
            }
            if metadata:
                message["metadata"] = metadata
            st.session_state.messages.append(message)
            st.rerun()
    
    def add_system_message(self, content: str, message_type: str = "info"):
        """Add a system message to the chat history (no notification)."""
        if content.strip():
            st.session_state.messages.append({
                "role": "system",
                "content": content,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_user_input(self) -> Optional[str]:
        """Get user input from the chat interface."""
        if prompt := st.chat_input("Ask a question about your documents..."):
            return prompt.strip()
        return None
    
    def display_system_status(self, rag_system):
        """Display the current system status."""
        if rag_system:
            status = rag_system.get_system_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", status['documents_loaded'])
            
            with col2:
                st.metric("Indexed", "✓" if status['is_indexed'] else "✗")
            
            with col3:
                st.metric("LLM Connected", "✓" if status['llm_connected'] else "✗")
            
            with col4:
                st.metric("Model", status['model_name'])
    
    def display_document_info(self):
        """Display information about processed documents."""
        if st.session_state.documents_processed:
            st.subheader("📄 Processed Documents")
            
            for doc_info in st.session_state.documents_processed:
                with st.expander(f"📄 {doc_info.get('file_name', 'Unknown file')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        chunks = doc_info.get('chunks_created', 0)
                        st.write(f"**Chunks:** {chunks}")
                    
                    with col2:
                        word_count = doc_info.get('word_count', 0)
                        st.write(f"**Words:** {word_count}")
                    
                    with col3:
                        success = doc_info.get('success', False)
                        status = '✓ Success' if success else '✗ Failed'
                        st.write(f"**Status:** {status}")
                    
                    if not success:
                        error = doc_info.get('error', 'Unknown error')
                        st.error(f"Error: {error}")
    
    def display_error_message(self, error: str):
        """Display an error message in the chat."""
        error_content = f"❌ **Error:** {error}"
        self.add_assistant_message(error_content)
        st.error(error)
    
    def display_success_message(self, message: str):
        """Display a success message in the chat."""
        success_content = f"✅ {message}"
        self.add_system_message(success_content, "success")
        st.success(message)
    
    def display_info_message(self, message: str):
        """Display an info message in the chat."""
        info_content = f"ℹ️ {message}"
        self.add_system_message(info_content, "info")
        st.info(message)
    
    def clear_chat_history(self):
        """Clear the chat message history."""
        st.session_state.messages = []
        st.rerun()
    
    def export_chat_history(self) -> str:
        """Export chat history as formatted text."""
        if not st.session_state.messages:
            return "No chat history to export."
        
        export_text = "# Chat History Export\n\n"
        export_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for message in st.session_state.messages:
            role = message["role"].title()
            content = message["content"]
            timestamp = message.get("timestamp", "")
            
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp
            else:
                time_str = ""
            
            export_text += f"## {role} ({time_str})\n\n{content}\n\n"
        
        return export_text
    
    def display_chat_controls(self):
        """Display chat control buttons."""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                self.clear_chat_history()
        
        with col2:
            if st.button("📥 Export Chat", help="Export chat history"):
                export_text = self.export_chat_history()
                st.download_button(
                    label="Download Chat History",
                    data=export_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
