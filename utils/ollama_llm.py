"""
Enhanced Ollama LLM Integration

This module provides a clean interface to Ollama LLM models for RAG applications.
It handles model interactions, response formatting, and error management.
"""

import requests
import json
from typing import Dict, Any, List, Optional
import time

class OllamaLLM:
    """
    Enhanced Ollama LLM client with robust error handling and response management.
    
    Features:
    - Clean API interface to Ollama
    - Automatic retry mechanisms
    - Response formatting and validation
    - Error handling and fallback responses
    """
    
    def __init__(self, model_name: str = "phi3", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama LLM client.

    Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {self.base_url}")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: The user's question or prompt
            context: Optional context information (e.g., retrieved documents)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not prompt.strip():
            return {
                'response': "I don't have a question to answer. Please provide a question.",
                'success': False,
                'error': 'Empty prompt'
            }
        
        # Build the full prompt with context if provided
        full_prompt = self._build_prompt(prompt, context)
        print(f"DEBUG: Final prompt sent to Ollama:\n{full_prompt[:1000]}\n---END PROMPT---")
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        # Attempt to generate response with retries
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                
                if 'response' in result:
                    return {
                        'response': result['response'].strip(),
                        'success': True,
                        'model': self.model_name,
                        'prompt_tokens': result.get('prompt_eval_count', 0),
                        'response_tokens': result.get('eval_count', 0),
                        'total_duration': result.get('total_duration', 0)
                    }
                else:
                    raise Exception("Invalid response format from Ollama")
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {
                        'response': "I'm taking too long to respond. Please try again.",
                        'success': False,
                        'error': 'timeout'
                    }
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request error on attempt {attempt + 1}: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {
                        'response': "I'm having trouble connecting. Please check if Ollama is running.",
                        'success': False,
                        'error': str(e)
                    }
                    
            except Exception as e:
                return {
                    'response': f"I encountered an error: {str(e)}",
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'response': "I'm unable to generate a response at the moment. Please try again.",
            'success': False,
            'error': 'max_retries_exceeded'
        }
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Build the full prompt with context if provided.
        
        Args:
            prompt: The user's question
            context: Optional context information
            
        Returns:
            Formatted prompt string
        """
        if context and context.strip():
            return f"""You are a helpful AI assistant that answers questions based on the provided document context. 
Use the information provided in the context to answer questions. If the context is not relevant to the question, do your best to summarize or describe the document content for the user.

Here is the document context:
{context}

Based on the above document context, please answer the following question:

Question: {prompt}

Answer (use the information from the context above, or summarize the document if the question is not directly answered):"""
        else:
            return f"""You are a helpful AI assistant. Please answer the following question:

Question: {prompt}

Answer:"""
    
    def chat_response(
        self, 
        messages: List[Dict[str, str]], 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat response based on conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context information
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not messages:
            return {
                'response': "No conversation history provided.",
                'success': False,
                'error': 'no_messages'
            }
        
        # Build conversation prompt
        conversation = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content.strip():
                conversation += f"{role.title()}: {content}\n"
        
        # Add context if provided
        if context:
            conversation = f"Context: {context}\n\n{conversation}"
        
        # Add the current prompt
        current_prompt = messages[-1].get('content', '') if messages else ''
        conversation += f"Assistant: "
        
        # Generate response
        return self.generate_response(conversation, context=None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get('models', [])
            for model in models:
                # Allow partial match for model name (ignore tags like :latest)
                if self.model_name in model.get('name', ''):
                    return {
                        'name': model.get('name'),
                        'size': model.get('size'),
                        'modified_at': model.get('modified_at'),
                        'available': True
                    }
            
            # If any model is available, return available True
            if models:
                return {'available': True, 'name': self.model_name}
            
            return {
                'name': self.model_name,
                'available': False,
                'error': 'Model not found'
            }

        except Exception as e:
            return {
                'name': self.model_name,
                'available': False,
                'error': str(e)
            }
