# LLM2KG: LLM Knowledge Graph construction and ReAct Agent for QA/Research
An end-to-end GraphRAG (Graph Retrieval-Augmented Generation) system that builds knowledge graphs from text documents and enables intelligent question answering with ReAct agents.

## Features

- **Agentic KG Construction**: Autonomous agent builds knowledge graphs with dynamic ontology extraction
- **ReAct QA Agent**: Reasoning + Acting agent for knowledge graph Q&A with hybrid retrieval
- **Uncertainty Metrics**: Objective confidence scoring (perplexity, semantic entropy, embedding consistency)
- **RAGAS Evaluation**: 3-layer evaluation framework with 8 metrics (no LLM-as-judge)
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
export GOOGLE_API_KEY=...          # Primary: RAGAS evaluation (Gemini 2.5 Pro)
export OPENAI_API_KEY=sk-...       # Fallback: RAGAS evaluation (GPT)
export TAVILY_API_KEY=tvly-...     # Optional: enables web search
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
| **Wikipedia Search** | `wiki_search_enabled` | True | Search Wikipedia for encyclopedic facts |
| **Web Search** | `web_search_enabled` | True | External search via Tavily API |
| **Auto Ingestion** | `auto_add_documents` | True | Adds web results to knowledge graph |

**Agent Tools:**
- `graph_lookup(entity_name)` - Look up entity and relationships
- `wiki_search(query)` - Search Wikipedia for encyclopedic information
- `web_search(query)` - Search the web (when enabled)
- `cypher_query(query)` - Execute Neo4j Cypher queries
- `finish(answer)` - Complete with final answer

**Tool Priority:** The agent prioritizes sources in order: Knowledge Graph → Wikipedia → Web Search

```python
# Minimal agent (graph lookup only)
agent = ReActQAAgent(
    use_retrieval_planning=False,
    compression_enabled=False,
    wiki_search_enabled=False,
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

### RAGAS-Based Evaluation (`benchmarks/agent_eval/`)

A 3-layer evaluation framework using RAGAS metrics (no LLM-as-judge):

| Layer | Metrics | Method |
|-------|---------|--------|
| **Retrieval** | Context Precision, Context Recall | RAGAS |
| **Agentic** | Loop Efficiency, Rejection Sensitivity | Formula-based |
| **Generation** | Faithfulness, Answer Relevancy, Answer Correctness, Factual Correctness | RAGAS |

> **Note:** Integrity layer disabled - all metrics used LLM-as-judge which has been removed.

```bash
# Run complete evaluation (8 test cases across 3 layers)
python benchmarks/run_complete_eval.py

# Ablation study with follow-up planning
python benchmarks/followup_ablation_study.py --quick

# Improved ablation study
python benchmarks/improved_ablation_study.py --study1 --test-run
```

### Uncertainty Metrics (`uncertainty_metrics.py`)

Objective confidence scoring replacing LLM self-reported confidence:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Perplexity** | Token probability via Ollama logprobs | Lower = more certain |
| **Semantic Entropy** | Consistency across multiple generations | Lower = more certain |
| **Embedding Consistency** | Cosine similarity of answer embeddings | Higher = more certain |
| **Combined Confidence** | Weighted average (40/30/30) | 0-1 scale |

```bash
# View detailed uncertainty scores
python agent_qa.py --question "What is inflation?" --verbose
```

### Ablation Study

Tests impact of each agent feature:

| Config | Description |
|--------|-------------|
| `baseline` | All features ON (default) |
| `no_planning` | Disable retrieval planning |
| `no_compression` | Disable context compression |
| `no_wiki` | Disable Wikipedia search |
| `no_web` | Disable web search |
| `no_auto_ingest` | Disable auto document ingestion |
| `followup_v*h*` | Follow-up question planning with configurable vector/hop limits |
| `minimal` | All features OFF |

**Key Insights:**
- Results vary significantly based on test case selection and knowledge graph content
- Simpler configurations often outperform feature-rich baseline on graph-focused queries
- Follow-up planning can improve multi-hop reasoning questions
- Run your own ablation study to find optimal config for your use case

## Project Structure

```
llm2kg/
├── agent_skb.py          # Knowledge graph construction agent
├── agent_qa.py           # ReAct QA agent
├── uncertainty_metrics.py # Confidence scoring (perplexity, entropy, consistency)
├── planned_graphrag.py   # CLaRa-style retrieval planning
├── ontologies.py         # Dynamic ontology extraction
├── graphrag.py           # GraphRAG retrieval utilities
├── skb_graphrag.py       # SKB-specific GraphRAG
├── frontend/
│   └── app.py            # Chainlit web application
├── prompts/              # LLM prompts and templates
├── benchmarks/
│   ├── agent_eval/       # RAGAS-based evaluation framework
│   │   ├── config.py     # Thresholds and LLM configuration
│   │   ├── runner.py     # Evaluation orchestrator
│   │   └── metrics/      # RAGAS + formula-based metrics
│   ├── run_complete_eval.py
│   ├── followup_ablation_study.py
│   └── improved_ablation_study.py
├── tests/                # Test suites
├── finetuning/           # SFT and DPO training pipelines
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
- Ollama with:
  - `nemotron-3-nano:30b` model (main inference)
  - `qwen3-embedding:8b` model (embeddings)
- Google API key (primary) or OpenAI API key (fallback) for RAGAS evaluation
- Tavily API key (optional, for web search)
- RAGAS package (`pip install ragas`)
