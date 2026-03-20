## 🎯 What this project does

1. Accepts natural language requests from ETL engineers (e.g., "Create a Lookup operator for joining customer_id to customer_dim").
2. Classifies intent via semantic embeddings and routes requests into two specialized processing branches:
   - `QA Node` for conceptual/design explanation.
   - `Config Specialist Node` for concrete JSON operator generation.
3. Enforces strict schema constraints and returns final JSON for downstream ETL platforms.
4. Tracks conversation history to support iterative refinements (e.g., "Change source key to order_id" afterwards).
5. Logs graph state transitions for full visibility in CI/CD or audit pipelines.

## 🌟 Why it matters

- Reduces manual configuration errors in ETL pipeline development.
- Accelerates build velocity for data engineering teams by 2x (demo metric).
- Enables non-expert data analysts to generate production-ready operator specs safely.
- Provides auditable and repeatable logic to satisfy governance requirements.

## 🧩 Tech stack & design highlights

- Python 3.12.10 (tested)
- LangGraph for agent orchestration and memory checkpointing
- Local HuggingFace `all-MiniLM-L6-v2` embeddings for data privacy
- Groq-hosted `llama-3.3-70b-versatile` for fast generative reasoning
- JSON Schema validation and strict output format for configuration safety
- CLI interface with optional stateful conversation storage

## 📁 Files in this repository

- `etl_assistant.py`: main entrypoint, graph node definitions, user loop
- `requirements.txt`: dependency list
- `etl_docs.txt`: domain docs + operators schema used by vector store
- `test_examples.txt`: sample queries to exercise capability cases
- `Writeup.pdf`: architecture + tradeoffs and evaluation notes

## ▶️ Quick start (for reviewers)

1. Install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Set `GROQ_API_KEY` in environment.
3. Run:
   ```bash
   python etl_assistant.py
   ```
4. Examples:
   - "Generate Lookup config for source table transactions and target table orders on order_id"
   - "What keys are required for a join lookup?"

## ✅ Outcome

- Easy-to-verify ETL config output JSON
- Two-stage pipeline for intent clarity and safety
- Built for production readiness with observability and schema checks
