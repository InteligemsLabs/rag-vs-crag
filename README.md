# RAG vs CRAG Comparison

A simple Streamlit application that demonstrates the differences between traditional **RAG (Retrieval-Augmented Generation)** and **CRAG (Corrective Retrieval-Augmented Generation)** systems.

## Overview

This project provides a side-by-side comparison of RAG and CRAG approaches:

- **RAG**: Traditional retrieval-augmented generation that retrieves documents and generates answers directly
- **CRAG**: Corrective RAG that evaluates document relevance, filters irrelevant content, and performs web search when needed

## Features

- 🔍 **Document Loading**: Load knowledge base from web URLs
- 📊 **Side-by-side Comparison**: Compare RAG and CRAG responses
- 🌐 **Web Search Integration**: CRAG performs web search when documents aren't relevant
- 📈 **Key Metrics Dashboard**: Clean metrics showing document usage, relevance, and performance
- 🎯 **Detailed Analytics**: View document evaluation, relevance scores, and system decisions
- 💡 **Sample Questions**: Pre-built questions to test both systems
- 🔍 **Smart Insights**: Insights about system performance differences

## Installation

```bash
git clone https://github.com/InteligemsLabs/rag-vs-crag.git
cd rag-vs-crag
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key_here          # For embeddings
OPENROUTER_API_KEY=your_openrouter_key_here  # For chat completions
TAVILY_API_KEY=your_tavily_key_here
```

## Usage

```bash
streamlit run app.py
```

1. Load documents (use default URLs or add your own)
2. Ask questions and compare RAG vs CRAG responses
3. View detailed analytics showing how CRAG makes decisions

## How It Works

### RAG System
1. Retrieves relevant documents from the vector store
2. Generates an answer using retrieved documents as context
3. Returns the response directly

### CRAG System
1. Retrieves initial documents from the vector store
2. **Evaluates** each document for relevance to the question
3. **Filters** out irrelevant documents
4. If relevance ratio is below threshold:
   - **Rewrites** the question for better search
   - **Performs web search** for additional information
5. Generates answer using filtered/enhanced documents

## Key Differences

| Feature | RAG | CRAG |
|---------|-----|------|
| Document Filtering | ❌ No | ✅ Yes |
| Relevance Evaluation | ❌ No | ✅ Yes |
| Web Search Fallback | ❌ No | ✅ Yes |
| Query Optimization | ❌ No | ✅ Yes |
| Error Correction | ❌ No | ✅ Yes |

## Example Use Cases

- **Information Retrieval**: Compare how both systems handle questions about your documents
- **Knowledge Base Gaps**: See how CRAG handles questions not covered in your knowledge base
- **Query Ambiguity**: Test both systems with unclear or poorly worded questions
- **Document Quality**: Observe how CRAG filters out irrelevant content

## Technical Details

### Dependencies
- **LangChain**: Framework for LLM applications
- **OpenAI**: Embeddings API
- **OpenRouter**: LLM API gateway for chat completions
- **ChromaDB**: Vector database for document storage
- **Tavily**: Web search API
- **Streamlit**: Web interface

### Configuration Options

Edit `config.py` to customize:
- Model selection (embeddings, chat model)
- Chunk size and overlap for document splitting
- Number of documents to retrieve
- Relevance threshold for CRAG
- Web search result count

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Based on the Corrective Retrieval Augmented Generation paper by Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling: https://arxiv.org/abs/2401.15884