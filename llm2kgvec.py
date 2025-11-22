import os
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI="bolt://localhost:7687"
#NEO4J_URI = os.environ["NEO4J_URI"]
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

# --- 2. ENHANCED NEO4J LOADER WITH VECTORS ---
class Neo4jVectorLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Initialize OpenAI Embeddings (Dimensions: 1536 for ada-002 / 3-small)
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    def close(self):
        self.driver.close()

    def sanitize(self, text):
        return re.sub(r'\W+', '_', text).upper()

    def init_indices(self):
        """Creates the Vector Index in Neo4j if it doesn't exist."""
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
            print("✅ Vector Index 'entity_embeddings' ready.")

    def add_graph_data(self, data: KnowledgeGraph):
        with self.driver.session() as session:
            # 1. Create Nodes with Embeddings
            for node in data.nodes:
                # We embed the text representation of the node (ID + Type)
                text_to_embed = f"{node.id} ({node.type})"
                vector = self.embeddings_model.embed_query(text_to_embed)
                
                session.execute_write(self._create_node_with_vector, node, vector)
            
            # 2. Create Edges (Standard)
            for edge in data.edges:
                session.execute_write(self._create_edge, edge)

    def _create_node_with_vector(self, tx, node: Node, vector: List[float]):
        # MERGE ensures no duplicates.
        # ON CREATE SET adds the embedding only when the node is first made.
        query = """
        MERGE (n:Entity {name: $name})
        ON CREATE SET n.type = $type, n.embedding = $vector
        """
        tx.run(query, name=node.id, type=node.type, vector=vector)

    def _create_edge(self, tx, edge: Edge):
        rel_type = self.sanitize(edge.relation)
        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[:{rel_type}]->(t)
        """
        tx.run(query, source=edge.source, target=edge.target)

    # --- NEW: HYBRID SEARCH FUNCTION ---
    def vector_search(self, query_text, limit=3):
        """Finds nodes semantically similar to the query text."""
        query_vector = self.embeddings_model.embed_query(query_text)
        
        # Cypher query to query the index
        cypher = """
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.type AS type, score
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, limit=limit, embedding=query_vector)
            return [record.data() for record in result]

# --- 3. PIPELINE EXECUTION ---
def run_graph_rag_pipeline():
    loader = Neo4jVectorLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Step 1: Setup Index
        loader.init_indices()
        
        # Step 2: Load Data (Simulated reuse of extraction logic)
        print("📥 Loading & Processing Data...")
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Setup Extraction Chain
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(KnowledgeGraph)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract nodes and edges."),
            ("human", "{text}"),
        ])
        chain = prompt | structured_llm

        # Limit for demo
        count = 0
        for entry in dataset:
            text = entry['text']
            if len(text.strip()) < 50: continue
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                try:
                    data = chain.invoke({"text": chunk})
                    if data.nodes:
                        loader.add_graph_data(data)
                        print(f"   ✅ Indexed {len(data.nodes)} nodes with vectors.")
                except Exception as e:
                    print(f"Error: {e}")
            
            count += 1
            if count >= 2: break # Stop after 2 articles for demo

        # Step 3: Perform a Vector Search
        print("\n🔎 TESTING VECTOR SEARCH:")
        print("-" * 30)
        # We search for a concept that might not match exact keywords
        search_term = "political leaders" 
        results = loader.vector_search(search_term)
        
        print(f"Query: '{search_term}'")
        for r in results:
            print(f"Found: {r['name']} (Type: {r['type']}) - Score: {r['score']:.4f}")

    finally:
        loader.close()

if __name__ == "__main__":
    run_graph_rag_pipeline()
