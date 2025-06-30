# 🤖 RAG Chat Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Streamlit application for document processing and AI-powered question answering using Retrieval-Augmented Generation (RAG) with Ollama. Transform your documents into an intelligent, searchable knowledge base.

## ✨ Features

### 📄 Document Processing
- **Multi-format Support**: PDF, DOCX, and TXT files
- **OCR Capabilities**: Extract text from scanned PDFs and images using Tesseract OCR
- **Batch Processing**: Handle multiple documents simultaneously
- **Intelligent Text Cleaning**: Automated preprocessing and normalization

### 🔍 Advanced Search
- **Hybrid Search Engine**: Combines dense embeddings (FAISS) with sparse TF-IDF retrieval
- **Semantic Chunking**: Smart text segmentation with configurable chunk size and overlap
- **High-Quality Embeddings**: Powered by Sentence Transformers

### 🤖 AI-Powered Responses
- **Context-Aware Answers**: Leverages Ollama's Phi3 model for intelligent responses
- **Source Citations**: Every answer includes relevant document references
- **Confidence Scores**: Transparent relevance scoring for retrieved content

### 💬 User Experience
- **Persistent Chat History**: Maintains conversation context across sessions
- **Session Management**: Easy data reset and chat clearing
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Real-time Processing**: Live status updates and error handling

## 🏗️ Project Structure

```
llm-Rag/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # MIT license file
└── utils/                   # Core functionality modules
    ├── __init__.py
    ├── rag.py               # RAG system orchestration
    ├── embedding.py         # Embedding generation & search
    ├── text_splitter.py     # Document chunking logic
    ├── pdf_reader.py        # PDF text extraction
    ├── docx_reader.py       # DOCX processing
    ├── ollama_llm.py        # Ollama LLM integration
    └── chat_interface.py    # Chat UI components
```

## 🚀 Quick Start

### Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://python.org/downloads/)
- **Ollama** - [Install Ollama](https://ollama.ai/download)
- **Tesseract OCR** - Required for PDF OCR functionality
  - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PawishrajhenAR/RAG-DocChatApp.git
   cd llm-Rag
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Linux/macOS)
   source venv/bin/activate
   
   # Activate (Windows)
   venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull the Phi3 model (in a new terminal)
   ollama pull phi3
   ```

### Launch the Application

```bash
streamlit run app.py
```

🎉 **Success!** Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar file uploader to select your documents
- Supported formats: PDF, DOCX, TXT
- Multiple files can be processed simultaneously

### 2. Process Documents
- Click **"Process Documents"** to extract and clean text
- The system will automatically handle OCR for scanned documents
- Monitor progress through real-time status updates

### 3. Build Search Index
- Click **"Build Search Index"** to create embeddings
- This enables semantic search across your document collection
- Index building may take a few moments for large documents

### 4. Start Chatting
- Use the chat interface to ask questions about your documents
- Receive contextual answers with source citations
- View confidence scores for retrieved information

### 5. Manage Your Session
- **Clear Chat**: Remove conversation history
- **Reset Data**: Start fresh with new documents
- **View Status**: Check processing and indexing status

## 🛠️ Configuration

### Customizing Chunk Settings

Modify chunking parameters in your configuration:

```python
# In utils/text_splitter.py
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
```

### Changing the LLM Model

Switch to different Ollama models:

```python
# In utils/ollama_llm.py
MODEL_NAME = "llama2"    # Or any other Ollama model
```

## 📊 API Usage

For programmatic access, use the RAG system directly:

```python
from utils.rag import RAGSystem

# Initialize the system
rag = RAGSystem(model_name="phi3")

# Process a document
rag.process_document("path/to/your/document.pdf")

# Build search index
rag.build_index()

# Ask questions
result = rag.ask_question("What are the main findings in this document?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## 🔧 Troubleshooting

### Common Issues

**🔴 Ollama Connection Error**
```
Solution: Ensure Ollama is running with `ollama serve`
Check if the model is available with `ollama list`
```

**🔴 Document Processing Failures**
```
Solution: Verify file format and ensure files aren't corrupted
Check file permissions and size limits
```

**🔴 OCR Not Working**
```
Solution: Install Tesseract OCR for your operating system
Verify installation with `tesseract --version`
```

**🔴 Slow Performance**
```
Solution: Reduce document size or decrease chunk size
Consider using a more powerful machine for large datasets
```

### Performance Tips

- **Optimize chunk size**: Smaller chunks = faster processing, larger chunks = better context
- **Use SSD storage**: Faster file I/O improves processing speed
- **Sufficient RAM**: Embedding generation is memory-intensive

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and test thoroughly**
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black utils/ app.py

# Lint code
flake8 utils/ app.py
```

## 📋 Requirements

### Python Dependencies
- `streamlit >= 1.28.0`
- `sentence-transformers >= 2.2.0`
- `faiss-cpu >= 1.7.4`
- `pypdf2 >= 3.0.0`
- `python-docx >= 0.8.11`
- `scikit-learn >= 1.3.0`
- `pytesseract >= 0.3.10`
- `pillow >= 10.0.0`
- `requests >= 2.31.0`

### System Dependencies
- Tesseract OCR
- Ollama with Phi3 model

## 🌟 Roadmap

- [ ] **Multi-language Support**: Add support for non-English documents
- [ ] **Cloud Integration**: AWS S3, Google Drive connectors
- [ ] **Advanced Analytics**: Document similarity analysis
- [ ] **Export Features**: Save conversations and insights
- [ ] **REST API**: Full API endpoints for integration
- [ ] **Docker Support**: Containerized deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the following open-source projects that make this possible:

- [**Ollama**](https://ollama.ai) - Local LLM inference
- [**Streamlit**](https://streamlit.io) - Web application framework
- [**Sentence Transformers**](https://www.sbert.net/) - Semantic embeddings
- [**FAISS**](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [**Tesseract OCR**](https://github.com/tesseract-ocr/tesseract) - Optical character recognition

## 👨‍💻 Author

**PAWISHRAJHEN A R**

- GitHub: [PawishrajhenAR](https://github.com/PawishrajhenAR)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/pawish6364/)
- Email: your.email@example.com

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

*Built with ❤️ using Python, Streamlit, and Ollama*

</div>
