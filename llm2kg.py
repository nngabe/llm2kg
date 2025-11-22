import os
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

# --- CONFIGURATION ---

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# --- 1. ONTOLOGY (Unchanged) ---
class Node(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'Albert Einstein'")
    type: str = Field(description="Category, e.g., 'Person', 'Location'")

class Edge(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(description="Relationship, e.g., 'born_in'")

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# --- 2. NEO4J CONNECTION MANAGER ---
class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def sanitize(self, text):
        # Convert "born in" -> "BORN_IN" for valid Cypher relationships
        return re.sub(r'\W+', '_', text).upper()

    def add_graph_data(self, data: KnowledgeGraph):
        with self.driver.session() as session:
            # 1. Create Nodes
            for node in data.nodes:
                session.execute_write(self._create_node, node)
            
            # 2. Create Edges
            for edge in data.edges:
                session.execute_write(self._create_edge, edge)

    def _create_node(self, tx, node: Node):
        # We use MERGE to avoid duplicates. 
        # We assign a generic label :Entity and a property 'type' 
        # to handle dynamic categories safely.
        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $type
        """
        tx.run(query, name=node.id, type=node.type)

    def _create_edge(self, tx, edge: Edge):
        # We sanitize the relationship type to prevent Cypher injection
        # and ensure valid syntax (e.g., :BORN_IN)
        rel_type = self.sanitize(edge.relation)
        
        # Cypher doesn't allow dynamic relationship types in parameters,
        # so we use f-string formatting safely after sanitization.
        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[:{rel_type}]->(t)
        """
        tx.run(query, source=edge.source, target=edge.target)

# --- 3. EXTRACTION LOGIC ---
def get_extraction_chain():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(KnowledgeGraph)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract nodes and edges. Use consistent IDs."),
        ("human", "{text}"),
    ])
    return prompt | structured_llm

# --- 4. MAIN PIPELINE ---
def process_to_neo4j(limit_samples=5):
    # Initialize Neo4j Driver
    neo4j_loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        print("📥 Loading Dataset (WikiText)...")
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chain = get_extraction_chain()

        count = 0
        print(f"🚀 Streaming to Neo4j (Limit: {limit_samples} articles)...")

        for entry in dataset:
            text = entry['text']
            if len(text.strip()) < 50: continue
            
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                try:
                    # Extract
                    graph_data = chain.invoke({"text": chunk})
                    
                    # Load to DB
                    if graph_data.nodes:
                        neo4j_loader.add_graph_data(graph_data)
                        print(f"   ✅ Inserted {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges.")
                except Exception as e:
                    print(f"   ❌ Error: {e}")

            count += 1
            if count >= limit_samples:
                break
                
    finally:
        neo4j_loader.close()
        print("🏁 Connection closed.")

if __name__ == "__main__":
    # Wait for Docker to be ready before running!
    process_to_neo4j(limit_samples=3)
