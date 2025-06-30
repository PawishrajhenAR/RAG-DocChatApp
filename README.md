# 🤖 RAG Chat Assistant

A powerful Streamlit application for document processing and AI-powered question answering using Retrieval-Augmented Generation (RAG) with Ollama. This application allows you to upload documents, process them, and get AI-powered answers based on the content of your documents.

## ✨ Key Features

### Core Features
- **Multi-format Document Support**: Upload and process PDF, DOCX, and TXT files with automatic format detection
- **Intelligent Text Processing**: Advanced text splitting with semantic chunking and cleaning
- **Hybrid Search**: Combines dense embeddings (FAISS) and sparse TF-IDF for optimal retrieval
- **AI-Powered Answers**: Uses Ollama's Phi3 model for accurate, context-aware responses
- **Document OCR**: Extract text from scanned PDFs and images using Tesseract OCR
- **Streaming Responses**: Real-time streaming of AI responses for better user experience
- **Session Management**: Persistent chat history and document state across sessions

### Advanced Features
- **Document Metadata Extraction**: Automatically extracts and indexes document metadata
- **Contextual Answers**: Provides source citations and confidence scores
- **Batch Processing**: Process multiple documents simultaneously
- **Configurable Chunking**: Customize text chunk size and overlap
- **Asynchronous Processing**: Non-blocking document processing
- **Error Handling**: Comprehensive error handling and user feedback
- **Responsive Design**: Works on both desktop and mobile devices

## 🏗️ Project Structure

```
llm-Rag/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Production dependencies
├── README.md              # This file
└── utils/                 # Core application modules
    ├── __init__.py
    ├── rag.py             # RAG system implementation
    ├── embedding.py       # Document embedding and search
    ├── text_splitter.py   # Intelligent text chunking
    ├── pdf_reader.py      # PDF text extraction
    ├── docx_reader.py    # Word document processing
    ├── ollama_llm.py     # Ollama LLM integration
    └── chat_interface.py # Chat UI management
```

## 🚀 Deployment

### Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Tesseract OCR installed (for PDF OCR support)
- At least 4GB RAM (8GB recommended for large documents)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-Rag
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - **Windows**: Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux (Ubuntu/Debian)**: `sudo apt install tesseract-ocr`

5. **Set up Ollama**
   - Download and install from [ollama.ai](https://ollama.ai)
   - Start the Ollama service
   - Pull the required model:
     ```bash
     ollama pull phi3
     ```

### Running in Production

1. **Start the application**
   ```bash
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

### Environment Variables

Configure the application using these environment variables:

```bash
# Ollama Configuration
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=phi3

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCS=100

# Optional: Enable debug mode
DEBUG=false
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running locally
3. **Phi3 model** pulled in Ollama

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-Rag
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv llm
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     llm\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source llm/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install and start Ollama**:
   - Download from [ollama.ai](https://ollama.ai)
   - Install and start the service
   - Pull the Phi3 model:
     ```bash
     ollama pull phi3
     ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload documents** using the sidebar file uploader

4. **Process documents** and build the search index

5. **Start chatting** with your documents!

## 📖 Usage Guide

### 1. Document Upload
- Use the sidebar to upload PDF, DOCX, or TXT files
- Multiple files can be uploaded simultaneously
- Click "Process Documents" to extract and chunk the text

### 2. Building the Search Index
- After processing documents, click "Build Search Index"
- This creates embeddings and enables document search
- The system uses hybrid search (dense + sparse) for better results

### 3. Asking Questions
- Type your questions in the chat interface
- The system will:
  - Search for relevant document chunks
  - Provide context to the AI
  - Generate accurate, contextual answers

### 4. Managing Your Session
- View system status in the sidebar
- Export chat history as Markdown
- Clear chat or all data as needed

## 🏗️ Architecture

The application is built with a modular architecture:

```
app.py                    # Main Streamlit application
├── utils/
│   ├── rag.py           # RAG system orchestration
│   ├── embedding.py     # Document embedding and search
│   ├── text_splitter.py # Intelligent text chunking
│   ├── pdf_reader.py    # PDF text extraction
│   ├── ollama_llm.py    # Ollama LLM integration
│   └── chat_interface.py # Chat UI management
```

### Key Components

- **RAGSystem**: Orchestrates the complete RAG pipeline
- **EnhancedEmbeddingIndex**: Hybrid search with FAISS and TF-IDF
- **EnhancedTextSplitter**: Intelligent document chunking
- **EnhancedPDFReader**: Robust PDF text extraction
- **OllamaLLM**: Clean Ollama API integration
- **ChatInterface**: Streamlit chat UI management

## 🔧 Configuration

### Model Settings
- Default model: `phi3`
- Ollama URL: `http://localhost:11434`
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

### Search Settings
- Hybrid search enabled by default
- Dense embedding model: `all-MiniLM-L6-v2`
- Maximum search results: 5 documents

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is pulled: `ollama list`
   - Verify the API endpoint is accessible

2. **Document Processing Failures**
   - Check file format support (PDF, DOCX, TXT)
   - Ensure files contain extractable text
   - Verify file permissions

3. **Search Index Build Failures**
   - Ensure documents were processed successfully
   - Check available memory for large documents
   - Verify sentence-transformers model download

4. **Slow Performance**
   - Reduce chunk size for faster processing
   - Use smaller documents for testing
   - Ensure adequate system resources

### Debug Information
The application provides detailed logging and status information:
- System status in the sidebar
- Processing progress indicators
- Error messages with context
- Document processing summaries

## 📝 API Reference

### RAGSystem Class
```python
rag = RAGSystem(model_name="phi3")

# Process documents
result = rag.process_document("path/to/document.pdf")

# Build search index
result = rag.build_index()

# Ask questions
result = rag.ask_question("What is the main topic?")
```

### ChatInterface Class
```python
chat = ChatInterface()

# Add messages
chat.add_user_message("Hello")
chat.add_assistant_message("Hi there!")

# Display interface
chat.display_chat_history()
## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for the local LLM infrastructure
- [Streamlit](https://streamlit.io) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [LangChain](https://langchain.com/) for text processing utilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Conclusion

This RAG Chat Assistant provides a powerful way to interact with your documents using natural language. With its modular architecture, it's easy to extend and customize for various use cases. Whether you're a researcher, student, or professional, this tool can help you quickly find information across multiple documents through an intuitive chat interface.

For support or feature requests, please open an issue on the GitHub repository.

---

Built using Python, Streamlit, and Ollama by PAWISHRAJHEN A R
