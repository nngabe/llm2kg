# LLM Knowledge Graph (KG) and Semi-structured Knowledge Bases (SKB) creation

This repo contains an end-to-end GraphRAG application in the following steps:
1. Pipeline for constructing a KG (as a Neo4J graph database) with vector indexing and node/edge text descriptions: `build_skb.py`
2. Entity resolution by similarity search on the KG vector index, with an LLM judge for final merging decisions: `resolve_entities.py`
3. Tools for GraphRAG using vector search + 1-hop neighbor enrichment:  `graphrag.py` uses only node and edge labels, `skb_graphrag.py` adds node/edge descriptions to context.
4. A CLI chatbot for answering questions based on a GraphRAG supplied context: `cli.py`.
5. Evaluation using six LLM-as-a-judge metrics: `eval.py`.

The KG can be built from text datasets on Economics, Law, and Physics found in `cais/wmdp-mmlu-auxiliary-corpora` [[1](https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora?library=datasets)].

## Setup

First, set up this repository: 
```
git clone https://github.com/nngabe/llm2kg.git
cd llm2kg/
```

Then start the docker containers for the Neo4J/Python environment (using configs in `docker-compose.yml`):
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
python build_skb.py --restart_index 0 --limit_docs 200 --subject economics
```
which indexes documents `0:200` with the parameters given.


Entity resolution can be performed to clean up the extracted graph with:

```
python resolve_entities.py 
```

The chatbot can be run with:

```
python cli.py
```
The chatbot is instructed only to answer questions using the retrieved context from the KG, and will not answer questions on other topics.

Lastly, a set of benchmark questions from Economics, Law, and Physics can be be used to evaluate the given Answer and Context from the GraphRAG enabled (context relevance, groundedness,   answer relevance, context precision, context recall, perplexity):

```
python eval.py
```
# Sample Responses
Here is a sample response from the chatbot building the KG and indexing documents `0:200` from the `economics` dataset:

## Sample Response 1
You > who are some famous physicists?   

Thinking... (Searching Knowledge Graph)

--- RETRIEVED CONTEXT ---
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

 **Chatbot >** The provided graph data does not include information about physicists. It focuses on economists and their contributions to economic theories and policies. If you have any questions related to the economists or economic concepts mentioned in the data, feel free to ask!

## Sample Response 2
You > who are some famous economists?

Thinking... (Searching Knowledge Graph)

--- RETRIEVED CONTEXT ---
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

-----------------------------

**Chatbot >** Based on the provided graph data, some famous economists include:

1. **John Maynard Keynes**: He is an influential economist known for his ideas on government intervention in the economy. His theories form the basis of Keynesian economics, which emphasizes the role of government intervention in managing economic cycles [John Maynard Keynes (Person)].

2. **Hyman Minsky**: Known for his work on financial instability and the nature of money, Minsky contributed significantly to the understanding of economic crises and the dynamics of financial markets [Hyman Minsky (Person)].

3. **George P. Schultz**: A professor of economics and former Secretary of the Treasury, Schultz has played a significant role in economic policy and management [George P. Schultz (Person)].

These individuals have made substantial contributions to economic theory and policy, influencing both academic thought and practical economic management.

## Sample Response 3
You > what are your favorite economic theories?

Thinking... (Searching Knowledge Graph)

--- RETRIEVED CONTEXT ---
  Retrieved Graph Data:
  
  === Entity: Keynesian Economists (Group) ===
  Summary: Economists who follow the theories of John Maynard Keynes, emphasizing the role of government intervention in the economy.
  Connections:
    -> [OPINION_ON] -> Macroeconomic Adjustment (Concept)
       Context: Keynesian economists argue that macroeconomic adjustment can take a very long time.
  
  === Entity: Modern Money Theory (Theory) ===
  Summary: An economic theory that suggests the state and the market are not opposing forces but are interconnected.
  Connections:
    -> [REDEFINES_RELATIONSHIP] -> State (Entity)
       Context: Modern Money Theory redefines the relationship between the state and the market.
    -> [REDEFINES_RELATIONSHIP] -> Market (Institution)
       Context: Modern Money Theory redefines the relationship between the market and the state.
  
  === Entity: Keynesian policy thinking (Economic Theory) ===
  Summary: An economic theory that emphasizes the role of government intervention and macroeconomic policy in managing economic cycles.
  Connections:
  
  === Entity: New Keynesian tradition (Economic Theory) ===
  Summary: An economic theory that attempts to reconcile rational expectations with the possibility and desirability of systematic macroeconomic policy.
  Connections:
    -> [RESPONSE_TO] -> Lucas model (Economic Model)
       Context: The New Keynesian tradition emerged as an attempt to address the limitations of the Lucas model by reconciling rational expectations with systematic policy.
       
-----------------------------

 **Chatbot >** As a helpful assistant, I don't have personal preferences or favorites. However, I can provide information on various economic theories based on the provided data:

1. **Keynesian Economics**: This theory, based on the ideas of John Maynard Keynes, emphasizes the role of government intervention in the economy. Keynesian economists argue that macroeconomic adjustment can take a very long time, and thus, active government policies are necessary to manage economic cycles and ensure stability [Keynesian Economists, Keynesian policy thinking].

2. **New Keynesian Economics**: This theory builds on Keynesian ideas by incorporating rational expectations and addressing the limitations of earlier models like the Lucas model. It emphasizes the importance of barriers to price adjustment and imperfect competition, suggesting that systematic macroeconomic policy can be both possible and desirable [New Keynesian tradition, New Keynesian thinking].

3. **Modern Money Theory (MMT)**: This theory suggests that the state and the market are interconnected rather than opposing forces. It redefines the relationship between the state and the market, proposing that governments can use their currency-issuing power to achieve full employment and control inflation [Modern Money Theory].

4. **Neoclassical Economics**: This perspective emphasizes the importance of aggregate supply and long-term productivity growth in determining potential GDP. Neoclassical economists believe in the natural adjustment of the economy to its potential GDP and typically advocate for a limited role of active stabilization policy, focusing instead on creating a stable economic environment [Neoclassical Perspective, Neoclassical Economists].

Each of these theories offers different insights and approaches to understanding and managing economic systems.

## Sample Evaluation
After building a KG only with Economics data, we can see that the GraphRAG system performs well on Economics questions, and very poorly on Law and Physics quesions, as expected. There does appear to be some transfer between the Economics we've constructed and the Law questions, but not the Physics questions, which makes sense.

`python eval.py`

--- ECON TEST REPORT ---
|    | Question                | Answer          |   Ctx_Rel. |   Ground. |   Answer_Rel. |   Ctx_Prec. |   Ctx_Rec. |   Perplx. |   **Comp_Score** |
|---:|:------------------------|:----------------|---------------:|---------------:|--------------:|----------------:|---------------:|-------------:|-------------:|
|  0 | Explain the concept ... | Definition      |           1    |           0.75 |          1    |            1    |           1    |         6.44 |         0.95 |
|  1 | What happens to the ... | When supply ... |           0.75 |           0.25 |          0.75 |            1    |           1    |         6.63 |         0.77 |
|  2 | Define Gross Domesti... | Definition:     |           1    |           1    |          0.75 |            0.75 |           1    |         6.91 |         0.91 |
|  3 | What is the differen... | Short answer... |           0.5  |           0.5  |          1    |            0.75 |           0.5  |         7.93 |         0.64 |
|  4 | How does an increase... | Short answer... |           0.75 |           0.75 |          0.75 |            0.25 |           0.5  |         7.33 |         0.59 |
|  5 | What describes a Mar... | A market mon... |           1    |           1    |          1    |            1    |           0.75 |         4.27 |         0.93 |
|  6 | What is Price Elasti... | Price elasti... |           1    |           1    |          0.75 |            1    |           1    |         3.85 |         0.95 |
|  7 | Explain the Law of D... | Definition      |           1    |           0.75 |          1    |            1    |           1    |         7.5  |         0.95 |
|  8 | What constitutes Fis... | Short answer... |           1    |           1    |          1    |            1    |           1    |         3.61 |         1    |
|  9 | What is the invisibl... | The graph do... |           0    |           0.75 |          1    |            0    |           0    |        10.74 |         0.32 |


  --- LAW TEST REPORT ---
|    | Question                | Answer          |   Ctx_Rel. |   Ground. |   Answer_Rel. |   Ctx_Prec. |   Ctx_Rec. |   Perplx. |   **Comp_Score** |
|---:|:------------------------|:----------------|---------------:|---------------:|--------------:|----------------:|---------------:|-------------:|-------------:|
|  0 | What are the four el... | I can’t find... |           0.5  |           0.5  |          1    |               0 |           0.5  |        18.13 |         0.5  |
|  1 | What is 'Considerati... | The graph da... |           0.5  |           0.75 |          1    |               0 |           0    |         7.97 |         0.41 |
|  2 | Explain the concept ... | I can’t find... |           0.25 |           0.75 |          0.25 |               0 |           0    |        13.66 |         0.23 |
|  3 | What does the First ... | The First Am... |           1    |           1    |          0.75 |               1 |           0.75 |        17.11 |         0.89 |
|  4 | What is the differen... | I cannot fin... |           0.5  |           0.5  |          0.75 |               0 |           0    |         9.09 |         0.32 |
|  5 | Explain the principl... | Short defini... |           0.5  |           0.5  |          1    |               0 |           0    |         8.32 |         0.36 |
|  6 | What is Double Jeopa... | The retrieve... |           0.25 |           1    |          0.25 |               0 |           0    |        23.14 |         0.27 |
|  7 | What is a Plaintiff?... | I can't find... |           0    |           1    |          0.25 |               0 |           0    |        14.2  |         0.23 |
|  8 | Define 'Habeas Corpu... | I can’t find... |           0.5  |           1    |          0.25 |               0 |           0    |        15.22 |         0.32 |
|  9 | What is the 'Statute... | I cannot fin... |           0.5  |           1    |          0.25 |               0 |           0    |        13.29 |         0.32 |


 --- PHYSICS TEST REPORT ---
|    | Question                | Answer          |   Ctx_Rel. |   Ground. |   Answ_Rel. |   Ctx_Prec. |   Ctx_Rec. |   Perplx. |   **Comp_Score** |
|---:|:------------------------|:----------------|---------------:|---------------:|--------------:|----------------:|---------------:|-------------:|-------------:|
|  0 | State Newton's Secon... | I’m unable t... |           0    |           1    |          0.25 |               0 |              0 |        14.79 |         0.23 |
|  1 | What is the differen... | I can’t answ... |           0    |           1    |          0.25 |               0 |              0 |        21.94 |         0.23 |
|  2 | Define Kinetic Energ... | I can’t find... |           0.25 |           0.25 |          0.75 |               0 |              0 |        23.36 |         0.23 |
|  3 | What does the First ... | I cannot ans... |           0    |           1    |          0.25 |               0 |              0 |        12.87 |         0.23 |
|  4 | Explain Ohm's Law....   | I can’t expl... |           0    |           1    |          0.25 |               0 |              0 |         9.16 |         0.23 |
|  5 | What is the accelera... | I can’t find... |           0.5  |           0.75 |          0.75 |               0 |              0 |        15.66 |         0.36 |
|  6 | Define Momentum....     | I can’t find... |           0    |           1    |          0.25 |               0 |              0 |        23.38 |         0.23 |
|  7 | What is Friction?...    | I cannot fin... |           0    |           0.75 |          0.75 |               0 |              0 |        16.66 |         0.27 |
|  8 | What is a Vector qua... | I don't have... |           0    |           0.75 |          0.25 |               0 |              0 |        19.7  |         0.18 |
|  9 | Explain the concept ... | The retrieve... |           0.25 |           1    |          0.25 |               0 |              0 |        12.24 |         0.27 |

