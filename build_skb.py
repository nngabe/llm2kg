import os
import re
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

# --- CONFIGURATION ---
# n.b. these are the defaults for neo4j inside the docker compose environment
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD= os.getenv("NEO4J_PASSWORD", "password")

# --- 1. ONTOLOGY (Semi-Structured) ---
class Node(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'Albert Einstein'")
    type: str = Field(description="Category, e.g., 'Person', 'Location'")
    # Capturing unstructured context
    description: str = Field(description="A brief summary of this entity based on the text.")

class Edge(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(description="Relationship, e.g., 'born_in'")
    # Capturing edge context
    description: str = Field(description="Context explaining why this relationship exists.")

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# --- 2. UPDATED LOADER ---
class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    def close(self):
        self.driver.close()

    def sanitize(self, text):
        return re.sub(r'\W+', '_', text).upper()

    def init_indices(self):
        # We index the node, but the vector will now represent id + description
        query = """
        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
        FOR (n:Entity)
        ON (n.embedding)
        OPTIONS {indexConfig: {
         `vector.dimensions`: 1536,
         `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    def add_graph_data(self, data: KnowledgeGraph):
        with self.driver.session() as session:
            # 1. Create Nodes with Rich Embeddings
            for node in data.nodes:
                # KEY CHANGE: We embed the description, not just the name.
                # This makes the vector search "Semantically Aware".
                text_to_embed = f"{node.id} ({node.type}): {node.description}"
                vector = self.embeddings_model.embed_query(text_to_embed)
                
                session.execute_write(self._create_node_with_metadata, node, vector)
            
            # 2. Create Edges with Description Properties
            for edge in data.edges:
                session.execute_write(self._create_edge_with_metadata, edge)

    def _create_node_with_metadata(self, tx, node: Node, vector: List[float]):
        # We store the description property on the node
        query = """
        MERGE (n:Entity {name: $name})
        ON CREATE SET 
            n.type = $type, 
            n.description = $desc,
            n.embedding = $vector
        """
        # Optional: ON MATCH SET n.description = $desc (if you want to overwrite)

        tx.run(query, name=node.id, type=node.type, desc=node.description, vector=vector)

    def _create_edge_with_metadata(self, tx, edge: Edge):
        rel_type = self.sanitize(edge.relation)
        
        # We store the description property on the relationship
        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r.description = $desc
        """
        tx.run(query, source=edge.source, target=edge.target, desc=edge.description)

# --- 3. UPDATED PIPELINE EXECUTION ---
def run_skb_pipeline(provider='openai', limit_docs=5, restart_index=0, subject='economics'):
    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        loader.init_indices()
        
        print(" Loading Data...")
        # (Assuming you are using the filter logic from the previous step, 
        # but simplified here for the example)
        if subject == 'economics':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "economics-corpus", split='train')
        elif subject == 'law':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "law-corpus", split='train')
        elif subject == 'physics':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "physics-corpus", split='train')

        dataset = dataset.select(range(restart_index,restart_index + limit_docs))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # --- KEY CHANGE: PROMPT ENGINEERING ---
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        structured_llm = llm.with_structured_output(KnowledgeGraph)
        
        system_prompt = """
        You are a Knowledge Graph expert. Extract a semi-structured graph from the text.
        
        1. Identify Entities (Nodes): Include a 'description' summarizing who/what the entity is.
        2. Identify Relationships (Edges): Include a 'description' explaining the context of the link.
        3. Use consistent IDs.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Text: {text}"),
        ])
        chain = prompt | structured_llm

        count = 0
        print(f'\n [doc][chunk]\n')
        for entry in dataset:
            text = entry['text']
            if len(text.strip()) < 50: continue
            
            chunks = text_splitter.split_text(text)
            
            for i,chunk in enumerate(chunks):
                try:
                    # The LLM now extracts descriptions automatically due to the updated Pydantic schema
                    data = chain.invoke({"text": chunk})
                    
                    if data.nodes:
                        loader.add_graph_data(data)
                        print(f" [{count+restart_index}][{i}] Indexed {len(data.nodes)} nodes with vector embedding and node/edge descriptions.")
                        
                        # DEBUG: Show what the LLM extracted to verify it's working
                        #print(f"      Example Node: {data.nodes[0].id} -> {data.nodes[0].description[:50]}...")
                        #print(f"      Example Edge: {data.edges[0].relation} -> {data.edges[0].description[:50]}...")
                        
                except Exception as e:
                    print(e)
            
            count += 1
            if count >= limit_docs: break 
    except Exception as e:
        print(e)

    finally:
        loader.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for LLM provider and KG pipeline config.")

    parser.add_argument("--provider", type=str, default='openai')
    parser.add_argument("--limit_docs", type=int, default=5)
    parser.add_argument("--restart_index", type=int, default=0)
    parser.add_argument("--subject", type=str, default='economics')

    kwargs = vars(parser.parse_args())
    print(f'kwargs: {kwargs}')

    providers_list = ['openai', 'google']
    subject_list = ['economics', 'law', 'physics']
    print(f'\nBuilding {kwargs["subject"]} KG with {kwargs["provider"]} vector index and text descriptions.\n')

    run_skb_pipeline(**kwargs)
