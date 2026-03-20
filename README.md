# Agentic RAG: ETL Operator Configuration Assistant

An intelligent, stateful CLI chatbot designed to assist ETL developers in configuring Lookup operators. Built using an Agentic AI workflow via LangGraph, this system dynamically routes user queries to specialized execution nodes to ensure highly accurate Q&A, strict JSON schema validation, and context-aware conversational memory.



## 🌟 Key Features
* **Agentic Intent Routing:** Employs a semantic classifier to route general conceptual questions to a `QA Node` and configuration tasks to a highly constrained `Config Specialist Node`.
* **Stateful Conversational Memory:** Utilizes LangGraph Checkpointers (`MemorySaver`) to maintain thread-level conversation history, allowing users to ask follow-up questions and dynamically modify generated JSON configurations.
* **Execution Observability:** Streams graph state updates to the terminal, providing full transparency into the agent's decision-making and routing process.
* **Local Embeddings for Privacy:** Uses HuggingFace's `all-MiniLM-L6-v2` to compute vector embeddings locally, ensuring sensitive ETL schemas never leave the environment.
* **Ultra-Low Latency Inference:** Powered by Meta's `llama-3.3-70b-versatile` model via the Groq API for rapid, accurate generation.

## 📋 Prerequisites
* [cite_start]**Python:** Version 3.8 or higher (tested on 3.12.10)[cite: 125].
* **API Key:** A valid [Groq API Key](https://console.groq.com/keys).

## 📁 Project Structure
* [cite_start]`etl_assistant.py` - The main Python script containing the LangGraph application and CLI interface.
* [cite_start]`requirements.txt` - List of required Python dependencies.
* [cite_start]`etl_docs.txt` - The source documentation and JSON schema loaded into the vector store[cite: 24, 82].
* [cite_start]`test_examples.txt` - A suite of test queries demonstrating the system's capabilities across different routing nodes[cite: 22].
* `Writeup.pdf` - A brief architectural overview detailing design decisions and trade-offs.

## 🚀 Setup Instructions

1. **Clone or Extract the Repository:**
   Ensure all project files are located in the same working directory.

2. **Create a Virtual Environment:**
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate