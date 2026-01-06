# GraphRAG Knowledge Base System

An end-to-end GraphRAG (Graph Retrieval-Augmented Generation) system that builds knowledge graphs from text documents and enables intelligent question answering with ReAct agents.

## Features

- **Agentic KG Construction**: Autonomous agent builds knowledge graphs with dynamic ontology extraction
- **ReAct QA Agent**: Reasoning + Acting agent for knowledge graph Q&A with hybrid retrieval
- **Enterprise Evaluation**: 4-layer evaluation framework with 14 metrics
- **Web Interface**: Interactive Chainlit app with graph visualization

## Quick Start

### 1. Setup

```bash
git clone https://github.com/nngabe/llm2kg.git
cd llm2kg/
docker compose up -d
docker compose exec llm-app bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY=sk-...       # Required for evaluation LLM judge
export TAVILY_API_KEY=tvly-...      # Optional: enables web search
```

### 3. Build Knowledge Graph

```bash
# Build KG from economics dataset (200 documents)
python agent_skb.py --subject economics --limit_docs 200

# Other subjects: law, physics
python agent_skb.py --subject law --limit_docs 100
```

### 4. Run the Application

**Option A: Web Interface (Recommended)**
```bash
chainlit run frontend/app.py --port 8000
```
Open http://localhost:8000

**Option B: Python API**
```python
from agent_qa import ReActQAAgent

agent = ReActQAAgent()
response = agent.answer_question("What is aggregate demand?")
print(response.answer)
agent.close()
```

## Architecture

### Knowledge Graph Construction (`agent_skb.py`)

The SKB (Semi-structured Knowledge Base) agent autonomously constructs knowledge graphs:

1. **Document Processing**: Ingests text documents from HuggingFace datasets
2. **Ontology Extraction**: Dynamically identifies entity types and relationships per document
3. **Entity Extraction**: Extracts entities and relationships using the ontology
4. **Graph Storage**: Stores in Neo4j with vector embeddings for similarity search

```bash
# Full options
python agent_skb.py --subject economics --limit_docs 200 --restart_index 0
```

### ReAct QA Agent (`agent_qa.py`)

The QA agent uses ReAct (Reasoning + Acting) for multi-step question answering:

| Feature | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **Retrieval Planning** | `use_retrieval_planning` | True | CLaRa-style entity/relationship planning |
| **Context Compression** | `compression_enabled` | True | Compresses observations to relevant facts |
| **Web Search** | `web_search_enabled` | True | External search via Tavily API |
| **Auto Ingestion** | `auto_add_documents` | True | Adds web results to knowledge graph |

**Agent Tools:**
- `graph_lookup(entity_name)` - Look up entity and relationships
- `cypher_query(query)` - Execute Neo4j Cypher queries
- `web_search(query)` - Search the web (when enabled)
- `finish(answer)` - Complete with final answer

```python
# Minimal agent (graph lookup only)
agent = ReActQAAgent(
    use_retrieval_planning=False,
    compression_enabled=False,
    web_search_enabled=False,
    auto_add_documents=False,
)
```

### Web Interface (`frontend/app.py`)

The Chainlit app provides three modes:

- **Classic Mode**: Traditional GraphRAG with entity extraction
- **Q&A Agent Mode**: Full ReAct agent with hybrid retrieval
- **Research Mode**: Autonomous gap-filling with approval workflow

Features:
- Chain-of-thought step visualization
- PyVis graph rendering
- Human-in-the-loop entity disambiguation

## Evaluation Framework

### Enterprise Evaluation (`benchmarks/enterprise_eval/`)

A comprehensive 4-layer evaluation framework:

| Layer | Purpose | Metrics |
|-------|---------|---------|
| **Retrieval** | Context quality | Contextual Precision, Contextual Recall, Graph Traversal Efficiency, Subgraph Connectivity |
| **Agentic** | Reasoning quality | Tool Selection Accuracy, Argument Correctness, Loop Efficiency, Rejection Sensitivity |
| **Integrity** | KG updates | Schema Adherence, Entity Disambiguation, Source Citation Accuracy |
| **Generation** | Answer quality | Faithfulness, Answer Relevance, Citation Recall |

```bash
# Run complete evaluation
python benchmarks/run_complete_eval.py

# Run ablation study
python benchmarks/enterprise_ablation_study.py

# Quick validation (4 test cases)
python benchmarks/enterprise_ablation_study.py --quick
```

### Ablation Study

Tests impact of each agent feature:

| Config | Description |
|--------|-------------|
| `baseline` | All features ON (default) |
| `no_planning` | Disable retrieval planning |
| `no_compression` | Disable context compression |
| `no_web` | Disable web search |
| `no_auto_ingest` | Disable auto document ingestion |
| `minimal` | All features OFF |

## Project Structure

```
llm2kg/
├── agent_skb.py          # Knowledge graph construction agent
├── agent_qa.py           # ReAct QA agent
├── build_skb.py          # Legacy KG builder
├── graphrag.py           # GraphRAG retrieval utilities
├── frontend/
│   └── app.py            # Chainlit web application
├── prompts/              # LLM prompts and templates
├── benchmarks/
│   ├── enterprise_eval/  # 4-layer evaluation framework
│   ├── run_complete_eval.py
│   └── enterprise_ablation_study.py
└── docker-compose.yml
```

## Data Sources

Knowledge graphs can be built from text datasets on:
- **Economics** - Economic concepts, theories, and policies
- **Law** - Legal terminology and case concepts
- **Physics** - Physical laws and scientific concepts

Source: [cais/wmdp-mmlu-auxiliary-corpora](https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora)

## Requirements

- Docker & Docker Compose
- Python 3.10+
- Neo4j (runs in container)
- Ollama with `nemotron-3-nano:30b` model (for local inference)
- OpenAI API key (for evaluation judge)
- Tavily API key (optional, for web search)
