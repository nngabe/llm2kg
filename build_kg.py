import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# --- 1. ECONOMICS ONTOLOGY ---

class EconEntityType(str, Enum):
    """
    Strict classification for Economic concepts and actors.
    """
    # Agents
    INSTITUTION = "Institution"      # e.g., Federal Reserve, IMF, World Bank
    AGENT = "Agent"                  # e.g., Consumers, Firms, Investors
    
    # Macro Concepts
    INDICATOR = "Indicator"          # e.g., GDP, Inflation Rate, Unemployment, CPI
    POLICY = "Policy"                # e.g., Quantitative Easing, Fiscal Stimulus
    CURRENCY = "Currency"            # e.g., USD, EUR, Gold
    
    # Micro/Market Concepts
    MARKET = "Market"                # e.g., Labor Market, Housing Market, Bond Market
    GOOD = "Good"                    # e.g., Oil, Semiconductor, Wheat
    CONCEPT = "Concept"              # e.g., Supply, Demand, Elasticity, Opportunity Cost

class EconRelationType(str, Enum):
    """
    Directional and Causal relationships are key in Economics.
    """
    # Causal / Directional
    INCREASES = "INCREASES"                 # Positive Causality (X goes up -> Y goes up)
    DECREASES = "DECREASES"                 # Negative Causality (X goes up -> Y goes down)
    STABILIZES = "STABILIZES"               # e.g., Policy -> Market
    
    # Structural
    CONTROLS = "CONTROLS"                   # e.g., Fed -> Interest Rates
    MEASURES = "MEASURES"                   # e.g., CPI -> Inflation
    TRADED_IN = "TRADED_IN"                 # e.g., Apple Stock -> Stock Market
    CONSUMES = "CONSUMES"                   # e.g., Households -> Goods

# --- 2. PYDANTIC SCHEMA ---

class EconNode(BaseModel):
    id: str = Field(description="Unique name. Use standard economic terms (e.g., 'Aggregate Demand').")
    type: EconEntityType = Field(description="The category of the concept.")
    description: str = Field(description="Definition or context of the concept.")

class EconEdge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relation: EconRelationType = Field(description="The causal or structural relationship.")
    description: str = Field(description="Explanation of the mechanism (e.g., 'Higher rates increase borrowing costs').")

class EconKnowledgeGraph(BaseModel):
    nodes: List[EconNode]
    edges: List[EconEdge]

# --- 3. LOADER ---

class Neo4jEconLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    def close(self):
        self.driver.close()

    def init_indices(self):
        with self.driver.session() as session:
            # 1. Vector Index for RAG
            session.run("""
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """)
            # 2. Indexes for fast traversal
            session.run("CREATE INDEX indicator_name IF NOT EXISTS FOR (n:Indicator) ON (n.name)")

    def add_data(self, data: EconKnowledgeGraph):
        with self.driver.session() as session:
            for node in data.nodes:
                vector = self.embeddings_model.embed_query(f"{node.id} ({node.type.value}): {node.description}")
                session.execute_write(self._merge_node, node, vector)
            for edge in data.edges:
                session.execute_write(self._merge_edge, edge)

    def _merge_node(self, tx, node: EconNode, vector: List[float]):
        label = node.type.value
        query = f"""
        MERGE (n:Entity {{name: $name}})
        SET n :{label}
        ON CREATE SET 
            n.type = $type_val,
            n.description = $desc,
            n.embedding = $vector
        """
        tx.run(query, name=node.id, type_val=label, desc=node.description, vector=vector)

    def _merge_edge(self, tx, edge: EconEdge):
        rel_type = edge.relation.value
        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r.description = $desc
        """
        tx.run(query, source=edge.source, target=edge.target, desc=edge.description)

# --- 4. EXECUTION ---

def process_econ_text():
    loader = Neo4jEconLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    loader.init_indices()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(EconKnowledgeGraph)
    
    system_prompt = """
    You are an Economics Professor building a causal knowledge graph.
    
    CRITICAL INSTRUCTION:
    Focus on CAUSALITY. Does X increase Y, or decrease Y?
    - If the text says "The Fed hiked rates to fight inflation,"
      Output: (Federal Reserve)-[CONTROLS]->(Interest Rates)
      Output: (Interest Rates)-[DECREASES]->(Inflation)
      
    Use the strict Ontology provided.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Text to analyze: {text}"),
    ])
    chain = prompt | structured_llm

    # TEXTBOOK EXAMPLE: The Phillips Curve / Monetary Policy
    text = """
    In response to overheating markets, the Federal Reserve implemented contractionary monetary policy.
    By raising the Federal Funds Rate, they increased the cost of borrowing for Firms.
    This leads to a decrease in Business Investment, which ultimately drives down Aggregate Demand.
    However, while this lowers Inflation, it typically increases Unemployment in the short run.
    """

    print("📈 Extracting Economic Models...")
    result = chain.invoke({"text": text})
    
    loader.add_data(result)
    
    # Verification
    print("\n--- Extracted Economic Relations ---")
    for e in result.edges:
        # Visualizing the causal chain
        arrow = "-->"
        if e.relation == "INCREASES": arrow = "++>"
        if e.relation == "DECREASES": arrow = "-->" # denotes negative
        
        print(f"{e.source} --[{e.relation.value}]--> {e.target}")
        print(f"   Mechanism: {e.description}")

    loader.close()

if __name__ == "__main__":
    process_econ_text()
