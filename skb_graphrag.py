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

class GraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def close(self):
        self.driver.close()

    def retrieve_graph_context(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves nodes by vector similarity, including their descriptions
        and the descriptions of their outgoing relationships.
        """
        query_vector = self.embeddings.embed_query(query_text)

        # UPDATED CYPHER QUERY
        cypher_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
        YIELD node AS head, score

        // Traverse 1 hop out
        OPTIONAL MATCH (head)-[r]->(tail)

        RETURN
            head.name AS entity,
            head.type AS type,
            head.description AS description,  // <--- NEW: Get Node Description
            score,
            COLLECT({
                relation: type(r),
                description: r.description,   // <--- NEW: Get Edge Description
                target: tail.name,
                target_type: tail.type
            }) AS relationships
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, limit=limit, embedding=query_vector)
            return [record.data() for record in result]

    def format_context(self, context_data: List[Dict]) -> str:
        """
        Constructs a rich text representation of the graph data.
        """
        formatted_text = "Retrieved Graph Data:\n"

        for entry in context_data:
            # 1. HEAD NODE CONTEXT
            # e.g., "Entity: Albert Einstein (Person)
            #        Summary: A theoretical physicist known for..."
            formatted_text += f"\n=== Entity: {entry['entity']} ({entry['type']}) ===\n"
            if entry['description']:
                formatted_text += f"Summary: {entry['description']}\n"

            # 2. RELATIONSHIP CONTEXT
            formatted_text += "Connections:\n"
            if entry['relationships']:
                for rel in entry['relationships']:
                    if rel['relation']:
                        # e.g., " -> [DEVELOPED] -> Theory of Relativity (Concept)"
                        #       "    Context: Published in 1905 during his Annus Mirabilis..."
                        formatted_text += f"  -> [{rel['relation']}] -> {rel['target']} ({rel['target_type']})\n"
                        if rel['description']:
                            formatted_text += f"     Context: {rel['description']}\n"
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

        # DEBUG: Print the rich context to see what the LLM sees
        # print(f"📄 Retrieved Context:\n{context_str}")

        # 3. Generate
        system_prompt = """
        You are a helpful assistant answering questions based *only* on the provided Knowledge Graph data.

        The graph data includes:
        - Entities (Nodes) with descriptions.
        - Connections (Edges) with context explaining the relationship.

        Use this rich context to provide a comprehensive answer.
        Cite the entities and relationships you used.
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
        # This question now benefits from the 'description' fields
        question = "Explain the connection between key political figures and their policies."

        answer = rag.generate_answer(question)

        print("\n🤖 LLM Answer:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
    finally:
        rag.close()
