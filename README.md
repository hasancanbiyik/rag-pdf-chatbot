# Simple RAG Chatbot

A basic Retrieval-Augmented Generation chatbot that answers questions about uploaded PDF documents using completely free, local tools.

## Quick Start

### 1. Install Ollama (Free Local LLM)
```bash
# Download from: https://ollama.ai
# Or use package manager:
# macOS: brew install ollama
# Windows: Download installer from website
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download AI Model
```bash
# Download the 3B model (about 2GB)
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

### 4. Run the App
```bash
# In a new terminal
streamlit run main.py
```

## How It Works

1. **PDF Upload**: User uploads a PDF document
2. **Text Extraction**: Extract text using PyPDF2
3. **Chunking**: Split text into manageable pieces
4. **Embeddings**: Convert chunks to vectors using SentenceTransformers
5. **Vector Storage**: Store in FAISS (Facebook AI Similarity Search)
6. **Query**: User asks a question
7. **Retrieval**: Find most relevant chunks
8. **Generation**: LLM generates answer based on context

## Key Features

- **100% Free** - No API costs
- **Local Processing** - Your data stays private
- **Source Citations** - Shows which parts of document were used
- **Conversation Memory** - Maintains chat history
- **Simple UI** - Easy to use Streamlit interface

## Project Structure
```
rag-chatbot/
├── main.py              # Main Streamlit app
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```

This project demonstrates:
- **AI/ML Skills**: RAG, embeddings, vector similarity
- **Backend**: Python, API integration, data processing
- **Frontend**: Streamlit, user interface design
- **System Design**: End-to-end AI application architecture

## Troubleshooting

**Ollama not connecting?**
- Make sure `ollama serve` is running
- Check if port 11434 is available

**Model too slow?**
- Try smaller model: `ollama pull llama3.2:1b`
- Reduce chunk size in the code

**Out of memory?**
- Use CPU-only FAISS: `faiss-cpu`
- Process smaller documents first

## Next Steps

1. Add more document types (Word, TXT)
2. Implement conversation history persistence
3. Add document management (multiple PDFs)
4. Deploy to Streamlit Cloud
5. Add evaluation metrics for answer quality
