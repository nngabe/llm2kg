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

Then start the docker containers for the Neo4J and Python environment with:
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

Here is a sample response from the chatbot after indexing documents `0:200` from the `economics` dataset:

## Sample Response 1
You > who are some famous physicists?   
🕷️  Thinking... (Searching Knowledge Graph)

--- 🔍 RETRIEVED CONTEXT ---
  Retrieved Graph Data:
  
  === Entity: Solow (Person) ===
  Summary: An economist known for his work on economic growth and the Solow residual.
  Connections:
    -> [COMPUTED] -> Solow residual (Economic Concept)
       Context: Solow computed the Solow residual for the U.S. economy.
  
  === Entity: Keynes (Person) ===
  Summary: An economist known for his contributions to monetary theory, including the concept of liquidity.
  Connections:
  
  === Entity: George P. Schultz (Person) ===
  Summary: A professor of economics and former Secretary of the Treasury and Director of the Office of Management and Budget.
  Connections:
    -> [IS_A] -> Economist (Profession)
       Context: George P. Schultz is a professor of economics, thus an economist.
       
-----------------------------

🔎 Searching Graph for: 'who are some famous physicists?'
🤖 Chatbot > The provided graph data does not include information about physicists. It focuses on economists and their contributions to economic theories and policies. If you have any questions related to the economists or economic concepts mentioned in the data, feel free to ask!

## Sample Response 2
You > who are some famous economists?
🕷️  Thinking... (Searching Knowledge Graph)

--- 🔍 RETRIEVED CONTEXT ---
  Retrieved Graph Data:
  
  === Entity: John Maynard Keynes (Person) ===
  Summary: An influential economist known for his ideas on government intervention in the economy.
  Connections:
    -> [INFLUENCES] -> Keynesian Economists (Group)
       Context: John Maynard Keynes' theories form the basis of Keynesian economics.
  
  === Entity: Keynesian Economists (Group) ===
  Summary: Economists who follow the theories of John Maynard Keynes, emphasizing the role of government intervention in the economy.
  Connections:
    -> [OPINION_ON] -> Macroeconomic Adjustment (Concept)
       Context: Keynesian economists argue that macroeconomic adjustment can take a very long time.
  
  === Entity: George P. Schultz (Person) ===
  Summary: A professor of economics and former Secretary of the Treasury and Director of the Office of Management and Budget.
  Connections:
    -> [IS_A] -> Economist (Profession)
       Context: George P. Schultz is a professor of economics, thus an economist.
    
    ...  

-----------------------------

🔎 Searching Graph for: 'who are some famous economists?'

🤖 Chatbot > Based on the provided graph data, some famous economists include:

1. **John Maynard Keynes**: He is an influential economist known for his ideas on government intervention in the economy. His theories form the basis of Keynesian economics, which emphasizes the role of government intervention in managing economic cycles [John Maynard Keynes (Person)].

2. **Hyman Minsky**: Known for his work on financial instability and the nature of money, Minsky contributed significantly to the understanding of economic crises and the dynamics of financial markets [Hyman Minsky (Person)].

3. **George P. Schultz**: A professor of economics and former Secretary of the Treasury, Schultz has played a significant role in economic policy and management [George P. Schultz (Person)].

These individuals have made substantial contributions to economic theory and policy, influencing both academic thought and practical economic management.
