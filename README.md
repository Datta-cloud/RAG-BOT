# RAGBook 📚

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload books/PDFs and ask questions about them.

## 🚀 Features

- 📄 Upload PDF books
- ✂️ Split documents into chunks
- 🔢 Generate embeddings using Mistral AI
- 🗄️ Store embeddings in ChromaDB
- 🔍 Retrieve relevant document chunks
- 🤖 Generate answers using Google Gemini
- 💬 Chat with your uploaded documents
- 🎯 MMR-based retrieval

## 🛠️ Technologies Used

- Python
- LangChain
- Streamlit
- Mistral AI
- Google Gemini
- ChromaDB
- PyPDF

## 🔄 RAG Workflow

PDF Upload  
↓  
Text Extraction  
↓  
Text Chunking  
↓  
Mistral Embeddings  
↓  
ChromaDB  
↓  
MMR Retriever  
↓  
Relevant Chunks  
↓  
Gemini  
↓  
Answer
