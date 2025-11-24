# LLM generation of Knowledge Graphs (KG) and Semi-structured Knowledge Bases (SKB)

This repo contains an end-to-end GraphRAG application in the following steps:
1. Pipeline for constructing a KG (as a Neo4J graph database) with vector indexing and node/edge text descriptions: `build_skb.py`
2. Utility to perform entity resolution by similarity search on the KG vector index, with an LLM judge for final merging decisions: `resolve_entities.py`
3. Tools for GraphRAG using similarity search and 1-hop neighbor enrichment:  `graphrag.py` uses only node and edge labels, `skb_graphrag.py` uses additional node/edge descriptions.
4. A CLI chatbot for answering questions based on a GraphRAG supplied context: `cli.py`.

The KG can be built from text datasets on economics, law, and physics found in `cais/wmdp-mmlu-auxiliary-corpora`[](https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora?library=datasets)

## Setup

First, set up this repository: 
```
git clone https://github.com/nngabe/llm2kg.git
cd llm2kg/
```

Then start the supplied docker containers for the Neo4J and Python environment with
```
docker compose exec llm-app bash
```

## Running the pipeline

The only environmental variable needed is an OpenAI API key, which you can set with
```
export OPENAI_API_KEY=sk-...
```
The Neo4j environment is already configured within the docker container, and the default URI, username, and password are used in all scripts. 
Note that the KG stored in the containers Neo4j instance will be lost when the container is closed.

The knowledge graph can be constructed from scratch (or restarted from index) with:

```
python build_skb.py --restart_index 0 --limit_docs 100 --subject economics
```
which indexes documents `0:100` with the parameters given.


Entity resolution can be performed to clean up the extracted graph with:

```
python resolve_entities.py 
```

Lastly, the chatbot can be run with:

```
python cli.py
```
