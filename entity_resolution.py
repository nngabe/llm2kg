import os
from typing import List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = "sk-..." 

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

class EntityResolver:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def close(self):
        self.driver.close()

    def find_potential_duplicates(self, similarity_threshold=0.90) -> List[Tuple[str, str, float]]:
        """
        Scans the graph for nodes with very similar vector embeddings.
        Returns a list of tuples: (Name A, Name B, Score)
        """
        print("🕵️ Scanning for duplicates using Vector Index...")
        
        # We iterate through nodes and query the index for neighbors.
        # In a massive production graph, you would use LSH (Locality Sensitive Hashing) here.
        # For <100k nodes, this approach is acceptable.
        query = """
        MATCH (n:Entity)
        CALL db.index.vector.queryNodes('entity_embeddings', 2, n.embedding)
        YIELD node AS candidate, score
        WHERE score > $threshold AND n.name <> candidate.name
        
        // Ensure we don't return (A, B) and (B, A) duplicates
        AND id(n) < id(candidate)
        
        RETURN n.name AS entity1, candidate.name AS entity2, score
        """
        
        duplicates = []
        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold)
            for record in result:
                duplicates.append((record["entity1"], record["entity2"], record["score"]))
        
        return duplicates

    def llm_check_is_same(self, name1, name2) -> bool:
        """
        Asks the LLM to act as a judge.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data cleaning expert. Determine if two entities refer to the same real-world object."),
            ("human", "Entity 1: {e1}\nEntity 2: {e2}\n\nAre these likely the same entity? Answer ONLY with 'YES' or 'NO'."),
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"e1": name1, "e2": name2}).content.strip().upper()
        
        return "YES" in response

    def merge_nodes(self, keep_name, discard_name):
        """
        Uses APOC to merge 'discard_node' INTO 'keep_node'.
        All relationships from 'discard_node' are moved to 'keep_node'.
        """
        print(f"   ⚡ Merging '{discard_name}' -> INTO -> '{keep_name}'")
        
        cypher = """
        MATCH (keep:Entity {name: $keep_name})
        MATCH (discard:Entity {name: $discard_name})
        
        // APOC mergeNodes logic:
        // nodes[0] is the target (keep), nodes[1] is the source (discard)
        CALL apoc.refactor.mergeNodes([keep, discard], {
            properties: {
                name: 'discard',      // keep the name of the first node
                embedding: 'discard', // keep the embedding of the first node
                type: 'overwrite'     // if types differ, take the second one (optional choice)
            },
            mergeRels: true
        })
        YIELD node
        RETURN count(*)
        """
        
        with self.driver.session() as session:
            session.run(cypher, keep_name=keep_name, discard_name=discard_name)

    def run_resolution_pipeline(self):
        # 1. Find Candidates
        candidates = self.find_potential_duplicates(similarity_threshold=0.92)
        
        print(f"Found {len(candidates)} pairs of potential duplicates.")
        
        for n1, n2, score in candidates:
            print(f"\nAnalyzing pair: '{n1}' vs '{n2}' (Score: {score:.4f})")
            
            # 2. LLM Verification
            is_match = self.llm_check_is_same(n1, n2)
            
            if is_match:
                print("   ✅ LLM Confirmed Match.")
                
                # 3. Heuristic: Keep the longer name (usually more descriptive)
                # e.g., Keep "John F. Kennedy", discard "J.F.K."
                if len(n1) >= len(n2):
                    keep, discard = n1, n2
                else:
                    keep, discard = n2, n1
                
                # 4. Execute Merge
                self.merge_nodes(keep, discard)
            else:
                print("   ❌ LLM Rejected Match (False Positive).")

if __name__ == "__main__":
    resolver = EntityResolver()
    try:
        resolver.run_resolution_pipeline()
    finally:
        resolver.close()
