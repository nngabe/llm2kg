"""
Tool data generator for agent SFT training.

Generates synthetic multi-turn conversations demonstrating tool use for GraphRAG.
Uses entities from the Neo4j knowledge graph and a teacher LLM to generate
realistic queries and tool calls.
"""

import os
import re
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

from neo4j import GraphDatabase
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# Scenario types for data generation
ScenarioType = Literal[
    "simple_lookup",      # Single graph_lookup
    "multi_hop",          # Sequential graph_lookups
    "disambiguation",     # entity_resolve -> graph_lookup
    "complex_query",      # cypher_query for aggregations
    "web_fallback",       # graph_lookup fails -> web_search
    "combined",           # Multiple tools in sequence
]


@dataclass
class ToolCall:
    """A single tool call in a conversation."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResponse:
    """Response from a tool call."""
    tool_call_id: str
    content: Dict[str, Any]


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: Literal["user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool responses


@dataclass
class ToolConversation:
    """A complete multi-turn conversation with tool use."""
    id: str
    scenario_type: ScenarioType
    turns: List[ConversationTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        turns = []
        for turn in self.turns:
            turn_dict = {"role": turn.role}
            if turn.content is not None:
                turn_dict["content"] = turn.content
            if turn.tool_calls:
                turn_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    }
                    for tc in turn.tool_calls
                ]
            if turn.tool_call_id:
                turn_dict["tool_call_id"] = turn.tool_call_id
            turns.append(turn_dict)

        return {
            "id": self.id,
            "scenario_type": self.scenario_type,
            "turns": turns,
            "metadata": self.metadata,
        }


class Neo4jSampler:
    """Sample entities and relationships from Neo4j for data generation."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def sample_entities(self, n: int = 10, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Sample random entities from the graph."""
        type_filter = "WHERE n.type = $entity_type" if entity_type else ""
        query = f"""
        MATCH (n:Entity)
        {type_filter}
        WITH n, rand() AS r
        ORDER BY r
        LIMIT $n
        RETURN n.name AS name, n.type AS type, n.description AS description
        """

        params = {"n": n}
        if entity_type:
            params["entity_type"] = entity_type

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def sample_entity_with_relationships(self, max_hops: int = 2) -> Dict[str, Any]:
        """Sample an entity with its relationships."""
        query = """
        MATCH (n:Entity)
        WHERE EXISTS { (n)-[]-() }
        WITH n, rand() AS r
        ORDER BY r
        LIMIT 1

        OPTIONAL MATCH path = (n)-[*1..2]-(related)
        WHERE related:Entity
        WITH n, collect(DISTINCT {
            name: related.name,
            type: related.type,
            description: related.description,
            path_length: length(path)
        })[0..10] AS relationships

        RETURN n.name AS name, n.type AS type, n.description AS description, relationships
        """

        with self.driver.session() as session:
            result = session.run(query, max_hops=max_hops)
            record = result.single()
            if record:
                return dict(record)
        return {}

    def get_entity_relationships(self, entity_name: str, max_hops: int = 1) -> Dict[str, Any]:
        """Get an entity's relationships."""
        query = """
        MATCH (n:Entity {name: $name})
        OPTIONAL MATCH (n)-[r]->(target:Entity)
        WITH n, collect({
            type: type(r),
            target: target.name,
            target_type: target.type,
            description: r.description
        }) AS outgoing

        OPTIONAL MATCH (source:Entity)-[r2]->(n)
        WITH n, outgoing, collect({
            type: type(r2),
            source: source.name,
            source_type: source.type,
            description: r2.description
        }) AS incoming

        RETURN n.name AS name, n.type AS type, n.description AS description,
               outgoing, incoming
        """

        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            record = result.single()
            if record:
                return dict(record)
        return {}

    def get_ambiguous_entities(self, n: int = 5) -> List[Dict[str, Any]]:
        """Find entities that could be ambiguous (similar names, common terms)."""
        query = """
        MATCH (n:Entity)
        WHERE size(n.name) < 30
        AND n.name =~ '(?i)^[A-Z][a-z]+ [A-Z][a-z]+.*'
        WITH n, rand() AS r
        ORDER BY r
        LIMIT $n
        RETURN n.name AS name, n.type AS type, n.description AS description
        """

        with self.driver.session() as session:
            result = session.run(query, n=n)
            return [dict(record) for record in result]

    def find_entities_by_partial_name(self, partial_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find entities matching a partial name for disambiguation."""
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($partial)
        RETURN n.name AS name, n.type AS type, n.description AS description
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, partial=partial_name, limit=limit)
            return [dict(record) for record in result]

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph for generating realistic queries."""
        query = """
        MATCH (n:Entity)
        WITH count(n) AS total_entities,
             collect(DISTINCT n.type) AS entity_types

        MATCH ()-[r]->()
        WITH total_entities, entity_types,
             count(r) AS total_relationships,
             collect(DISTINCT type(r))[0..20] AS relationship_types

        RETURN total_entities, entity_types, total_relationships, relationship_types
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return dict(record)
        return {}

    def run_cypher_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]


class ToolDataGenerator:
    """
    Generate synthetic tool-use conversations for agent SFT.

    Uses entities from Neo4j and a teacher LLM to create diverse
    training examples covering different tool-use scenarios.
    """

    def __init__(
        self,
        neo4j_sampler: Optional[Neo4jSampler] = None,
        teacher_llm=None,
        seed: int = 42,
    ):
        self.sampler = neo4j_sampler or Neo4jSampler()
        self.teacher_llm = teacher_llm
        self.rng = random.Random(seed)
        self._call_counter = 0

    def close(self):
        if self.sampler:
            self.sampler.close()

    def _generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        self._call_counter += 1
        return f"call_{self._call_counter}"

    def _generate_conversation_id(self, scenario_type: str) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = self.rng.randint(1000, 9999)
        return f"{scenario_type}_{timestamp}_{rand_suffix}"

    # --- Scenario Generators ---

    def generate_simple_lookup(self) -> Optional[ToolConversation]:
        """Generate a simple single-tool lookup scenario."""
        # Sample an entity with relationships
        entity = self.sampler.sample_entity_with_relationships()
        if not entity or not entity.get("name"):
            return None

        name = entity["name"]
        entity_type = entity.get("type", "Entity")
        description = entity.get("description", "No description available.")
        relationships = entity.get("relationships", [])

        # Generate user question
        questions = [
            f"What do you know about {name}?",
            f"Can you tell me about {name}?",
            f"Who is {name}?" if entity_type == "Person" else f"What is {name}?",
            f"Give me information about {name}.",
            f"I need details on {name}.",
        ]
        user_question = self.rng.choice(questions)

        # Create tool call
        call_id = self._generate_call_id()
        tool_call = ToolCall(
            id=call_id,
            name="graph_lookup",
            arguments={
                "entity_name": name,
                "include_relationships": True,
                "max_hops": 1,
            }
        )

        # Create tool response
        tool_response = {
            "entity": {
                "name": name,
                "type": entity_type,
                "description": description,
            },
            "relationships": [
                {
                    "type": rel.get("type", "RELATED_TO") if isinstance(rel, dict) else "RELATED_TO",
                    "target": rel.get("name", "Unknown") if isinstance(rel, dict) else str(rel),
                    "target_type": rel.get("type", "Entity") if isinstance(rel, dict) else "Entity",
                }
                for rel in relationships[:5]
            ] if relationships else [],
        }

        # Generate assistant response based on tool results
        rel_text = ""
        if tool_response["relationships"]:
            rel_parts = [
                f"{rel['type'].replace('_', ' ').lower()} {rel['target']}"
                for rel in tool_response["relationships"][:3]
            ]
            rel_text = f" This entity is connected to: {', '.join(rel_parts)}."

        assistant_response = (
            f"Based on the knowledge graph, {name} is a {entity_type.lower()}. "
            f"{description[:200]}{'...' if len(description) > 200 else ''}{rel_text}"
        )

        # Build conversation
        turns = [
            ConversationTurn(role="user", content=user_question),
            ConversationTurn(role="assistant", tool_calls=[tool_call]),
            ConversationTurn(
                role="tool",
                content=json.dumps(tool_response),
                tool_call_id=call_id
            ),
            ConversationTurn(role="assistant", content=assistant_response),
        ]

        return ToolConversation(
            id=self._generate_conversation_id("simple_lookup"),
            scenario_type="simple_lookup",
            turns=turns,
            metadata={"entity": name, "entity_type": entity_type},
        )

    def generate_multi_hop(self) -> Optional[ToolConversation]:
        """Generate a multi-hop query scenario with sequential lookups."""
        # Sample entity with relationships
        entity = self.sampler.sample_entity_with_relationships()
        if not entity or not entity.get("relationships"):
            return None

        name = entity["name"]
        relationships = entity.get("relationships", [])

        if not relationships:
            return None

        # Pick a related entity for second hop
        related = self.rng.choice(relationships)
        related_name = related.get("name") if isinstance(related, dict) else str(related)

        if not related_name:
            return None

        # Generate user question asking about both
        questions = [
            f"How is {name} connected to {related_name}?",
            f"What is the relationship between {name} and {related_name}?",
            f"Tell me about {name} and its connection to {related_name}.",
        ]
        user_question = self.rng.choice(questions)

        turns = [
            ConversationTurn(role="user", content=user_question),
        ]

        # First lookup
        call_id_1 = self._generate_call_id()
        tool_call_1 = ToolCall(
            id=call_id_1,
            name="graph_lookup",
            arguments={"entity_name": name, "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_1]))

        # First response
        first_response = {
            "entity": {"name": name, "type": entity.get("type", "Entity"), "description": entity.get("description", "")},
            "relationships": [{"target": related_name, "type": "RELATED_TO"}],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(first_response), tool_call_id=call_id_1))

        # Second lookup for related entity
        call_id_2 = self._generate_call_id()
        tool_call_2 = ToolCall(
            id=call_id_2,
            name="graph_lookup",
            arguments={"entity_name": related_name, "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_2]))

        # Second response
        second_response = {
            "entity": {
                "name": related_name,
                "type": related.get("type", "Entity") if isinstance(related, dict) else "Entity",
                "description": related.get("description", "") if isinstance(related, dict) else "",
            },
            "relationships": [],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(second_response), tool_call_id=call_id_2))

        # Final response
        assistant_response = (
            f"Based on the knowledge graph, {name} is connected to {related_name}. "
            f"{name} is described as: {entity.get('description', 'No description')[:150]}... "
            f"And {related_name} is a related entity in the knowledge graph."
        )
        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        return ToolConversation(
            id=self._generate_conversation_id("multi_hop"),
            scenario_type="multi_hop",
            turns=turns,
            metadata={"entities": [name, related_name]},
        )

    def generate_disambiguation(self) -> Optional[ToolConversation]:
        """Generate an entity disambiguation scenario."""
        # Sample ambiguous entities
        entities = self.sampler.get_ambiguous_entities(n=10)
        if not entities:
            entities = self.sampler.sample_entities(n=10)

        if not entities:
            return None

        entity = self.rng.choice(entities)
        name = entity["name"]

        # Create a partial/ambiguous name
        name_parts = name.split()
        if len(name_parts) >= 2:
            ambiguous_name = name_parts[0]  # Just first name/word
        else:
            ambiguous_name = name[:len(name)//2]  # First half of name

        # Find similar entities for disambiguation candidates
        candidates = self.sampler.find_entities_by_partial_name(ambiguous_name, limit=5)
        if len(candidates) < 2:
            candidates = [entity, {"name": f"{name} (alternative)", "type": entity.get("type"), "description": "Alternative entity"}]

        # Generate question with ambiguous reference
        questions = [
            f"What can you tell me about {ambiguous_name}?",
            f"Who is {ambiguous_name}?",
            f"I'm looking for information on {ambiguous_name}.",
        ]
        user_question = self.rng.choice(questions)

        turns = [
            ConversationTurn(role="user", content=user_question),
        ]

        # Entity resolve call
        call_id_1 = self._generate_call_id()
        tool_call_1 = ToolCall(
            id=call_id_1,
            name="entity_resolve",
            arguments={
                "entity_name": ambiguous_name,
                "context": "",
                "max_candidates": 5,
            },
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_1]))

        # Resolve response with candidates
        resolve_response = {
            "candidates": [
                {
                    "name": c.get("name", "Unknown"),
                    "type": c.get("type", "Entity"),
                    "score": round(0.95 - i * 0.1, 2),
                }
                for i, c in enumerate(candidates[:3])
            ],
            "best_match": entity["name"],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(resolve_response), tool_call_id=call_id_1))

        # Now do a graph lookup on the resolved entity
        call_id_2 = self._generate_call_id()
        tool_call_2 = ToolCall(
            id=call_id_2,
            name="graph_lookup",
            arguments={"entity_name": entity["name"], "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_2]))

        # Lookup response
        lookup_response = {
            "entity": {
                "name": entity["name"],
                "type": entity.get("type", "Entity"),
                "description": entity.get("description", ""),
            },
            "relationships": [],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(lookup_response), tool_call_id=call_id_2))

        # Final response mentioning disambiguation
        candidate_names = [c.get("name") for c in candidates[:3] if c.get("name")]
        assistant_response = (
            f"The name '{ambiguous_name}' could refer to several entities: "
            f"{', '.join(candidate_names)}. "
            f"Based on context, I found information about {entity['name']}: "
            f"{entity.get('description', 'No description available')[:150]}..."
        )
        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        return ToolConversation(
            id=self._generate_conversation_id("disambiguation"),
            scenario_type="disambiguation",
            turns=turns,
            metadata={"ambiguous_name": ambiguous_name, "resolved_entity": entity["name"]},
        )

    def generate_complex_query(self) -> Optional[ToolConversation]:
        """Generate a complex Cypher query scenario."""
        # Get graph stats for realistic queries
        stats = self.sampler.get_graph_stats()
        entity_types = stats.get("entity_types", ["Person", "Organization", "Concept"])
        rel_types = stats.get("relationship_types", ["RELATED_TO"])

        if not entity_types:
            entity_types = ["Entity"]

        entity_type = self.rng.choice(entity_types)

        # Generate complex questions and matching Cypher
        query_templates = [
            {
                "question": f"How many {entity_type}s are in the knowledge graph?",
                "cypher": f"MATCH (n:Entity) WHERE n.type = '{entity_type}' RETURN count(n) AS count",
                "result_key": "count",
            },
            {
                "question": f"What are the most connected {entity_type}s?",
                "cypher": f"""MATCH (n:Entity)-[r]-() WHERE n.type = '{entity_type}'
                             RETURN n.name AS name, count(r) AS connections
                             ORDER BY connections DESC LIMIT 5""",
                "result_key": "top_entities",
            },
            {
                "question": "Which entities have the most relationships?",
                "cypher": """MATCH (n:Entity)-[r]-()
                            RETURN n.name AS name, n.type AS type, count(r) AS relationships
                            ORDER BY relationships DESC LIMIT 10""",
                "result_key": "top_connected",
            },
        ]

        template = self.rng.choice(query_templates)
        user_question = template["question"]
        cypher = template["cypher"]

        turns = [
            ConversationTurn(role="user", content=user_question),
        ]

        # Cypher query call
        call_id = self._generate_call_id()
        tool_call = ToolCall(
            id=call_id,
            name="cypher_query",
            arguments={"query": cypher, "params": {}},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call]))

        # Execute query and get response
        try:
            results = self.sampler.run_cypher_query(cypher)
            query_response = {"records": results[:10], "count": len(results)}
        except Exception as e:
            query_response = {"records": [], "count": 0, "error": str(e)}

        turns.append(ConversationTurn(role="tool", content=json.dumps(query_response), tool_call_id=call_id))

        # Generate response based on results
        if query_response.get("records"):
            records = query_response["records"]
            if "count" in str(records[0]) if records else "":
                count_val = records[0].get("count", 0) if records else 0
                assistant_response = f"Based on my analysis, there are {count_val} {entity_type}s in the knowledge graph."
            else:
                names = [r.get("name", "Unknown") for r in records[:5]]
                assistant_response = f"The query returned these results: {', '.join(names)}."
        else:
            assistant_response = "The query did not return any results. The knowledge graph may not have data matching this criteria."

        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        return ToolConversation(
            id=self._generate_conversation_id("complex_query"),
            scenario_type="complex_query",
            turns=turns,
            metadata={"cypher": cypher, "entity_type": entity_type},
        )

    def generate_web_fallback(self) -> Optional[ToolConversation]:
        """Generate a scenario where graph lookup fails and web search is used."""
        # Use a made-up entity that won't be in the graph
        fake_entities = [
            "QuantumAI Corporation",
            "Project Nexus Initiative",
            "Dr. Alexandra Chen",
            "The 2024 Global Summit",
            "NeuraTech Industries",
        ]
        fake_entity = self.rng.choice(fake_entities)

        questions = [
            f"What do you know about {fake_entity}?",
            f"Can you find information on {fake_entity}?",
            f"Tell me about {fake_entity}.",
        ]
        user_question = self.rng.choice(questions)

        turns = [
            ConversationTurn(role="user", content=user_question),
        ]

        # First try graph lookup
        call_id_1 = self._generate_call_id()
        tool_call_1 = ToolCall(
            id=call_id_1,
            name="graph_lookup",
            arguments={"entity_name": fake_entity, "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_1]))

        # Empty response (not found)
        empty_response = {"entity": None, "relationships": [], "message": "Entity not found in knowledge graph"}
        turns.append(ConversationTurn(role="tool", content=json.dumps(empty_response), tool_call_id=call_id_1))

        # Fallback to web search
        call_id_2 = self._generate_call_id()
        tool_call_2 = ToolCall(
            id=call_id_2,
            name="web_search",
            arguments={"query": fake_entity, "num_results": 5},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_2]))

        # Web search response (simulated)
        web_response = {
            "results": [
                {"title": f"About {fake_entity}", "snippet": f"Information about {fake_entity}...", "url": "https://example.com/1"},
                {"title": f"{fake_entity} - Overview", "snippet": "Overview and details...", "url": "https://example.com/2"},
            ]
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(web_response), tool_call_id=call_id_2))

        # Final response acknowledging the fallback
        assistant_response = (
            f"I couldn't find {fake_entity} in the knowledge graph, so I searched the web. "
            f"Based on web results, {fake_entity} appears to be a topic with information available online, "
            f"though I don't have detailed knowledge graph data about it."
        )
        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        return ToolConversation(
            id=self._generate_conversation_id("web_fallback"),
            scenario_type="web_fallback",
            turns=turns,
            metadata={"entity": fake_entity, "fallback_used": True},
        )

    def generate_combined(self) -> Optional[ToolConversation]:
        """Generate a complex scenario using multiple tools."""
        # Sample entities
        entities = self.sampler.sample_entities(n=3)
        if len(entities) < 2:
            return None

        entity1 = entities[0]
        entity2 = entities[1]

        # Complex question requiring multiple tools
        questions = [
            f"Compare {entity1['name']} and {entity2['name']} based on the knowledge graph.",
            f"What do {entity1['name']} and {entity2['name']} have in common?",
            f"How are {entity1['name']} and {entity2['name']} related in the knowledge base?",
        ]
        user_question = self.rng.choice(questions)

        turns = [
            ConversationTurn(role="user", content=user_question),
        ]

        # First entity lookup
        call_id_1 = self._generate_call_id()
        tool_call_1 = ToolCall(
            id=call_id_1,
            name="graph_lookup",
            arguments={"entity_name": entity1["name"], "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_1]))

        response_1 = {
            "entity": {"name": entity1["name"], "type": entity1.get("type", "Entity"), "description": entity1.get("description", "")},
            "relationships": [],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(response_1), tool_call_id=call_id_1))

        # Second entity lookup
        call_id_2 = self._generate_call_id()
        tool_call_2 = ToolCall(
            id=call_id_2,
            name="graph_lookup",
            arguments={"entity_name": entity2["name"], "include_relationships": True, "max_hops": 1},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_2]))

        response_2 = {
            "entity": {"name": entity2["name"], "type": entity2.get("type", "Entity"), "description": entity2.get("description", "")},
            "relationships": [],
        }
        turns.append(ConversationTurn(role="tool", content=json.dumps(response_2), tool_call_id=call_id_2))

        # Cypher query to find connections
        call_id_3 = self._generate_call_id()
        cypher = f"""
        MATCH path = shortestPath((a:Entity {{name: '{entity1['name']}'}})-[*..3]-(b:Entity {{name: '{entity2['name']}'}}))
        RETURN [node IN nodes(path) | node.name] AS path_nodes, length(path) AS path_length
        """
        tool_call_3 = ToolCall(
            id=call_id_3,
            name="cypher_query",
            arguments={"query": cypher.strip(), "params": {}},
        )
        turns.append(ConversationTurn(role="assistant", tool_calls=[tool_call_3]))

        response_3 = {"records": [], "message": "No direct path found between entities"}
        turns.append(ConversationTurn(role="tool", content=json.dumps(response_3), tool_call_id=call_id_3))

        # Final comprehensive response
        assistant_response = (
            f"Based on my analysis of the knowledge graph:\n\n"
            f"**{entity1['name']}** ({entity1.get('type', 'Entity')}): {entity1.get('description', 'No description')[:100]}...\n\n"
            f"**{entity2['name']}** ({entity2.get('type', 'Entity')}): {entity2.get('description', 'No description')[:100]}...\n\n"
            f"I searched for direct connections between these entities but didn't find a direct path. "
            f"They may be related through other entities not directly connected."
        )
        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        return ToolConversation(
            id=self._generate_conversation_id("combined"),
            scenario_type="combined",
            turns=turns,
            metadata={"entities": [entity1["name"], entity2["name"]]},
        )

    # --- Main Generation Method ---

    def generate_conversation(self, scenario_type: Optional[ScenarioType] = None) -> Optional[ToolConversation]:
        """
        Generate a synthetic conversation for the given scenario type.

        Args:
            scenario_type: Type of scenario to generate, or None for random

        Returns:
            ToolConversation or None if generation failed
        """
        if scenario_type is None:
            scenario_type = self.rng.choice([
                "simple_lookup",
                "multi_hop",
                "disambiguation",
                "complex_query",
                "web_fallback",
                "combined",
            ])

        generators = {
            "simple_lookup": self.generate_simple_lookup,
            "multi_hop": self.generate_multi_hop,
            "disambiguation": self.generate_disambiguation,
            "complex_query": self.generate_complex_query,
            "web_fallback": self.generate_web_fallback,
            "combined": self.generate_combined,
        }

        generator = generators.get(scenario_type)
        if generator:
            try:
                return generator()
            except Exception as e:
                logger.warning(f"Failed to generate {scenario_type}: {e}")
                return None
        return None

    def generate_dataset(
        self,
        num_samples: int = 1000,
        scenario_distribution: Optional[Dict[ScenarioType, float]] = None,
    ) -> List[ToolConversation]:
        """
        Generate a dataset of synthetic conversations.

        Args:
            num_samples: Total number of conversations to generate
            scenario_distribution: Dict mapping scenario types to probabilities

        Returns:
            List of ToolConversation objects
        """
        if scenario_distribution is None:
            scenario_distribution = {
                "simple_lookup": 0.30,
                "multi_hop": 0.20,
                "disambiguation": 0.15,
                "complex_query": 0.15,
                "web_fallback": 0.10,
                "combined": 0.10,
            }

        # Normalize distribution
        total = sum(scenario_distribution.values())
        scenario_distribution = {k: v / total for k, v in scenario_distribution.items()}

        # Calculate samples per type
        samples_per_type = {}
        remaining = num_samples
        for scenario_type, prob in scenario_distribution.items():
            count = int(num_samples * prob)
            samples_per_type[scenario_type] = count
            remaining -= count

        # Distribute remaining samples
        for scenario_type in samples_per_type:
            if remaining <= 0:
                break
            samples_per_type[scenario_type] += 1
            remaining -= 1

        # Generate conversations
        conversations = []
        for scenario_type, count in samples_per_type.items():
            logger.info(f"Generating {count} {scenario_type} conversations...")
            for _ in range(count):
                conv = self.generate_conversation(scenario_type)
                if conv:
                    conversations.append(conv)

        logger.info(f"Generated {len(conversations)} total conversations")
        return conversations
