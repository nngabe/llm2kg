import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI = os.environ["NEO4J_URI"] 
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DBNAME = "wikitext-small"

class GraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def close(self):
        self.driver.close()

    def retrieve_graph_context(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        1. Vector Search to find relevant nodes.
        2. Traversal to find immediate neighbors (1-hop).
        """
        query_vector = self.embeddings.embed_query(query_text)
        
        # This Cypher query does two things:
        # 1. Finds the top 'limit' nodes by vector similarity.
        # 2. OPTIONALLY matches any outgoing relationships from those nodes to get context.
        cypher_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
        YIELD node AS head, score
        
        // Traverse 1 hop out to gather context (Who is this node related to?)
        OPTIONAL MATCH (head)-[r]->(tail)
        
        // Return the data in a clean dictionary format
        RETURN 
            head.name AS entity, 
            head.type AS type, 
            score, 
            COLLECT({relation: type(r), target: tail.name, target_type: tail.type}) AS relationships
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, limit=limit, embedding=query_vector)
            return [record.data() for record in result]

    def format_context(self, context_data: List[Dict]) -> str:
        """Converts the list of dictionaries into a readable string for the LLM."""
        formatted_text = "Graph Data:\n"
        for entry in context_data:
            formatted_text += f"- Entity: {entry['entity']} ({entry['type']})\n"
            if entry['relationships']:
                for rel in entry['relationships']:
                    if rel['relation']: # Check if relationship exists
                        formatted_text += f"  -> {rel['relation']} -> {rel['target']} ({rel['target_type']})\n"
            else:
                formatted_text += "  (No outgoing relationships found)\n"
        return formatted_text

    def generate_answer(self, user_question: str):
        print(f"🔎 Searching Graph for: '{user_question}'...")
        
        # 1. Retrieve
        context_data = self.retrieve_graph_context(user_question)
        
        if not context_data:
            return "I couldn't find any relevant information in the Knowledge Graph."

        # 2. Format
        context_str = self.format_context(context_data)
        print(f"📄 Retrieved Context:\n{context_str}")

        # 3. Generate (The RAG Prompt)
        system_prompt = """
        You are a helpful assistant answering questions based *only* on the provided Knowledge Graph data.
        - If the answer is not in the graph data, say "I don't know."
        - Cite the entities and relationships you used to form your answer.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context_str, 
            "question": user_question
        })
        
        return response

# --- EXECUTION ---
if __name__ == "__main__":
    rag = GraphRAG()
    
    try:
        # Example Question
        # (Assuming you ran the previous ingestion scripts)
        question = "Tell me about the political figures and their roles."
        
        answer = rag.generate_answer(question)
        
        print("\n🤖 LLM Answer:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
    finally:
        rag.close()
