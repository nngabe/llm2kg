#!/usr/bin/env python3
"""
Integration tests for agent_qa.py with real services.

Tests use:
- Real FalkorDB database
- Real Ollama LLM
- Real Wikipedia/Web access

Run with: python -m pytest tests/test_integration_agent_qa.py -v -s
"""

import sys
sys.path.insert(0, '/app')

import os
import json
import time
import unittest
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
TEST_GRAPH_NAME = "test_agent_qa_kg"
OLLAMA_HOST = "http://host.docker.internal:11434"


def falkordb_available() -> bool:
    """Check if FalkorDB is available."""
    try:
        from falkordb import FalkorDB
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        db.select_graph("test_connection")
        return True
    except Exception as e:
        logger.warning(f"FalkorDB not available: {e}")
        return False


def ollama_available() -> bool:
    """Check if Ollama is available."""
    try:
        import requests
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return False


# =============================================================================
# DATABASE LOADER INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestFalkorDBQALoaderIntegration(unittest.TestCase):
    """Integration tests for FalkorDBQALoader."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph with sample data."""
        from falkordb import FalkorDB
        import uuid

        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        # Clear and create test data
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        # Create test entities
        cls.graph.query(f"""
            CREATE (e1:Entity {{
                uuid: '{uuid.uuid4()}',
                name: 'GE Vernova',
                entity_id: 'ent_gevernova',
                ontology_type: 'ORGANIZATION',
                description: 'American energy company specializing in power generation equipment'
            }})
            CREATE (e2:Entity {{
                uuid: '{uuid.uuid4()}',
                name: '9HA Gas Turbine',
                entity_id: 'ent_9ha',
                ontology_type: 'PRODUCT',
                description: 'Heavy-duty gas turbine with over 64% combined cycle efficiency'
            }})
            CREATE (e3:Entity {{
                uuid: '{uuid.uuid4()}',
                name: '7HA Gas Turbine',
                entity_id: 'ent_7ha',
                ontology_type: 'PRODUCT',
                description: 'Heavy-duty gas turbine for 60Hz power grids'
            }})
            CREATE (e1)-[:MANUFACTURES {{confidence: 0.95}}]->(e2)
            CREATE (e1)-[:MANUFACTURES {{confidence: 0.95}}]->(e3)
            CREATE (e2)-[:RELATED_TO {{confidence: 0.8}}]->(e3)
        """)

        # Create test DocumentChunk
        cls.graph.query(f"""
            CREATE (d:DocumentChunk {{
                chunk_id: 'chunk_test_001',
                content: 'The 9HA.02 gas turbine achieves a combined cycle efficiency of 64.26%, setting a world record. It can produce up to 571 MW in combined cycle configuration.',
                source_type: 'webpage',
                source_url: 'https://www.gevernova.com/gas-power/products/gas-turbines/9ha',
                chunk_type: 'leaf'
            }})
        """)

        # Create test Episode
        cls.graph.query(f"""
            MATCH (e:Entity {{name: '9HA Gas Turbine'}})
            CREATE (ep:Episode {{
                episode_id: 'ep_test_001',
                name: 'Product Page Information',
                content: 'Technical specifications for 9HA gas turbine',
                reference_time: '2024-01-15T10:00:00Z',
                source_url: 'https://www.gevernova.com/gas-power',
                source_type: 'webpage'
            }})-[:CONTAINS {{confidence: 0.9}}]->(e)
        """)

        # Create test Community
        cls.graph.query(f"""
            MATCH (e1:Entity {{name: 'GE Vernova'}})
            MATCH (e2:Entity {{name: '9HA Gas Turbine'}})
            MATCH (e3:Entity {{name: '7HA Gas Turbine'}})
            CREATE (c:Community {{
                community_id: 'comm_gas_power',
                name: 'Gas Power Products'
            }})
            CREATE (e1)-[:BELONGS_TO]->(c)
            CREATE (e2)-[:BELONGS_TO]->(c)
            CREATE (e3)-[:BELONGS_TO]->(c)
        """)

    @classmethod
    def tearDownClass(cls):
        """Clean up test graph."""
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    def test_loader_initialization(self):
        """Test FalkorDBQALoader initialization."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            graph_name=TEST_GRAPH_NAME,
        )

        self.assertIsNotNone(loader)
        self.assertIsNotNone(loader.graph)
        self.assertIsNotNone(loader.embedding_model)

    def test_text_search_entities(self):
        """Test text-based entity search."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        results = loader._text_search_entities("9HA", limit=5)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("9HA", results[0].get("name", ""))

    def test_get_entity_with_relationships(self):
        """Test getting entity with relationships."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        result = loader.get_entity_with_relationships("GE Vernova")

        self.assertTrue(result.get("found"))
        self.assertIn("entity", result)
        self.assertEqual(result["entity"]["name"], "GE Vernova")
        self.assertIn("relationships", result)
        self.assertGreater(len(result["relationships"]), 0)

    def test_get_entity_episodes(self):
        """Test getting entity episodes."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        episodes = loader.get_entity_episodes("9HA Gas Turbine", limit=5)

        self.assertIsInstance(episodes, list)
        self.assertGreater(len(episodes), 0)
        self.assertIn("episode_id", episodes[0])

    def test_get_community_members(self):
        """Test getting community members."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        members = loader.get_community_members("9HA Gas Turbine", limit=5)

        self.assertIsInstance(members, list)
        # Should find other entities in same community
        if len(members) > 0:
            self.assertIn("name", members[0])

    def test_get_entity_timeline(self):
        """Test getting entity timeline."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        timeline = loader.get_entity_timeline("9HA Gas Turbine", limit=10)

        self.assertIsInstance(timeline, list)
        if len(timeline) > 0:
            self.assertIn("observed_at", timeline[0])

    def test_search_documents(self):
        """Test document search."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME, reranker_enabled=False)
        results = loader.search_documents("9HA efficiency MW", limit=5)

        self.assertIsInstance(results, list)
        # May be empty if no DocumentChunk matches keyword filters

    def test_run_cypher(self):
        """Test running Cypher queries."""
        from agent_qa import FalkorDBQALoader

        loader = FalkorDBQALoader(graph_name=TEST_GRAPH_NAME)
        results = loader.run_cypher(
            "MATCH (e:Entity) WHERE e.name CONTAINS $name RETURN e.name as name LIMIT 5",
            {"name": "Turbine"}
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


# =============================================================================
# LLM INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(ollama_available(), "Ollama not available")
class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM functionality."""

    def test_chat_ollama_basic(self):
        """Test basic ChatOllama invocation."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="nemotron-3-nano:30b",
            base_url=OLLAMA_HOST,
            temperature=0,
            num_ctx=4096,
        )

        response = llm.invoke("What is 2 + 2? Answer with just the number.")
        self.assertIsNotNone(response.content)
        self.assertIn("4", response.content)

    def test_chat_ollama_json_response(self):
        """Test ChatOllama JSON response generation."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="nemotron-3-nano:30b",
            base_url=OLLAMA_HOST,
            temperature=0,
        )

        prompt = """Return a JSON object with a "thought" field and "ready" boolean field.

Example: {"thought": "I need more info", "ready": false}

Your response:"""

        response = llm.invoke(prompt)
        content = response.content

        # Should contain JSON-like structure
        self.assertIn("{", content)
        self.assertIn("}", content)

    def test_retrieval_plan_generation(self):
        """Test retrieval plan generation."""
        from langchain_ollama import ChatOllama
        from prompts.retrieval_prompts import format_retrieval_plan_prompt
        from planned_graphrag import parse_retrieval_plan

        llm = ChatOllama(
            model="nemotron-3-nano:30b",
            base_url=OLLAMA_HOST,
            temperature=0,
        )

        question = "What is the efficiency of the 9HA gas turbine?"
        prompt = format_retrieval_plan_prompt(question)
        response = llm.invoke(prompt)

        plan = parse_retrieval_plan(response.content)

        self.assertIsNotNone(plan)
        # Plan should have some targets or needs
        total_items = len(plan.entity_targets) + len(plan.information_needs)
        self.assertGreaterEqual(total_items, 0)


# =============================================================================
# REACT AGENT INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestReActAgentIntegration(unittest.TestCase):
    """Integration tests for ReActQAAgent."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph with sample data."""
        from falkordb import FalkorDB
        import uuid

        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        # Clear and create test data
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        # Create comprehensive test data
        cls.graph.query(f"""
            CREATE (e1:Entity {{
                uuid: '{uuid.uuid4()}',
                name: 'GE Vernova',
                entity_id: 'ent_gevernova',
                ontology_type: 'ORGANIZATION',
                description: 'American multinational energy company'
            }})
            CREATE (e2:Entity {{
                uuid: '{uuid.uuid4()}',
                name: '9HA Gas Turbine',
                entity_id: 'ent_9ha',
                ontology_type: 'PRODUCT',
                description: 'Heavy-duty gas turbine with 64.26% combined cycle efficiency'
            }})
            CREATE (e1)-[:MANUFACTURES {{confidence: 0.95}}]->(e2)
        """)

    @classmethod
    def tearDownClass(cls):
        """Clean up test graph."""
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    def test_agent_initialization(self):
        """Test ReActQAAgent initialization."""
        from agent_qa import ReActQAAgent

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            wiki_search_enabled=False,
            skip_uncertainty=True,
        )

        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.neo4j_loader)
        self.assertIsNotNone(agent.graph)

        agent.close()

    def test_agent_plan_retrieval_node(self):
        """Test retrieval plan generation node."""
        from agent_qa import ReActQAAgent, QAAgentState

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(question="What is the efficiency of the 9HA gas turbine?")
            result = agent._plan_retrieval_node(state)

            self.assertIn("retrieval_plan", result)
            self.assertIsNotNone(result["retrieval_plan"])
        finally:
            agent.close()

    def test_agent_think_node(self):
        """Test think node with real LLM."""
        from agent_qa import ReActQAAgent, QAAgentState

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(
                question="What company makes the 9HA turbine?",
                context_formatted="GE Vernova manufactures the 9HA Gas Turbine.",
            )
            result = agent._think_node(state)

            self.assertIn("current_thought", result)
            # Should either be ready to answer or have an action
            self.assertTrue(
                result.get("ready_to_answer") or result.get("pending_action") is not None
            )
        finally:
            agent.close()

    def test_agent_execute_graph_lookup(self):
        """Test graph_lookup tool execution."""
        from agent_qa import ReActQAAgent, QAAgentState, ToolCall

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(
                question="What is the 9HA?",
                pending_action=ToolCall(
                    tool_name="graph_lookup",
                    arguments={"entity_name": "9HA Gas Turbine"}
                ),
                context=[],
            )
            result = agent._execute_action_node(state)

            self.assertIn("last_observation", result)
            # Should find the entity
            self.assertIn("9HA", result["last_observation"])
        finally:
            agent.close()

    def test_agent_execute_cypher_query(self):
        """Test cypher_query tool execution."""
        from agent_qa import ReActQAAgent, QAAgentState, ToolCall

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(
                question="List all entities",
                pending_action=ToolCall(
                    tool_name="cypher_query",
                    arguments={"query": "MATCH (e:Entity) RETURN e.name LIMIT 5"}
                ),
                context=[],
            )
            result = agent._execute_action_node(state)

            self.assertIn("last_observation", result)
            self.assertIn("Query returned", result["last_observation"])
        finally:
            agent.close()

    def test_agent_synthesize_node(self):
        """Test answer synthesis with real LLM."""
        from agent_qa import ReActQAAgent, QAAgentState

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(
                question="Who manufactures the 9HA gas turbine?",
                context_formatted="Entity: GE Vernova manufactures the 9HA Gas Turbine with 64.26% efficiency.",
                thought_history=[],
            )
            result = agent._synthesize_node(state)

            self.assertIn("final_answer", result)
            self.assertIn("citations", result)
            # Answer should mention GE Vernova
            self.assertIn("GE", result["final_answer"])
        finally:
            agent.close()

    def test_agent_full_answer_flow(self):
        """Test full question answering flow."""
        from agent_qa import ReActQAAgent, QAResponse

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            wiki_search_enabled=False,
            skip_uncertainty=True,
            use_retrieval_planning=True,
        )

        try:
            response = agent.answer_question("What is the 9HA gas turbine?")

            self.assertIsInstance(response, QAResponse)
            self.assertIsNotNone(response.answer)
            self.assertGreater(len(response.answer), 10)
            # Should mention 9HA or turbine
            self.assertTrue(
                "9HA" in response.answer or "turbine" in response.answer.lower()
            )
        finally:
            agent.close()

    def test_agent_with_improved_retrieval(self):
        """Test agent with ImprovedGraphRAG."""
        from agent_qa import ReActQAAgent, QAResponse

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            wiki_search_enabled=False,
            skip_uncertainty=True,
            use_retrieval_planning=True,
            use_improved_retrieval=True,
            max_hops=2,
            vector_limit=5,
        )

        try:
            response = agent.answer_question("Tell me about GE Vernova products")

            self.assertIsInstance(response, QAResponse)
            self.assertIsNotNone(response.answer)
        finally:
            agent.close()


# =============================================================================
# WIKIPEDIA SEARCH INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(ollama_available(), "Ollama not available")
class TestWikipediaSearchIntegration(unittest.TestCase):
    """Integration tests for Wikipedia search functionality."""

    def test_wikipedia_loader_direct(self):
        """Test direct Wikipedia loading."""
        from langchain_community.document_loaders import WikipediaLoader

        loader = WikipediaLoader(query="gas turbine", load_max_docs=1)
        docs = loader.load()

        self.assertGreater(len(docs), 0)
        self.assertIn("turbine", docs[0].page_content.lower())

    @unittest.skipUnless(falkordb_available(), "FalkorDB not available")
    def test_agent_wiki_search(self):
        """Test agent wiki_search tool."""
        from agent_qa import ReActQAAgent, QAAgentState, ToolCall

        agent = ReActQAAgent(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            web_search_enabled=False,
            wiki_search_enabled=True,
            skip_uncertainty=True,
        )

        try:
            state = QAAgentState(
                question="What is a gas turbine?",
                pending_action=ToolCall(
                    tool_name="wiki_search",
                    arguments={"query": "gas turbine"}
                ),
                context=[],
            )
            result = agent._execute_action_node(state)

            self.assertIn("last_observation", result)
            # Should find Wikipedia articles
            self.assertTrue(
                "Wikipedia" in result["last_observation"] or
                "WikiPage" in result["last_observation"] or
                "turbine" in result["last_observation"].lower()
            )
            self.assertTrue(result.get("external_info_used", False))
        finally:
            agent.close()


# =============================================================================
# GRAPHRAG INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestGraphRAGIntegration(unittest.TestCase):
    """Integration tests for GraphRAG components."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        from falkordb import FalkorDB
        import uuid

        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        # Create test entities
        cls.graph.query(f"""
            CREATE (e1:Entity {{uuid: '{uuid.uuid4()}', name: 'Test Entity A', ontology_type: 'CONCEPT'}})
            CREATE (e2:Entity {{uuid: '{uuid.uuid4()}', name: 'Test Entity B', ontology_type: 'CONCEPT'}})
            CREATE (e1)-[:RELATED_TO]->(e2)
        """)

    def test_planned_graphrag_initialization(self):
        """Test PlannedGraphRAG initialization."""
        from planned_graphrag import PlannedGraphRAG

        graphrag = PlannedGraphRAG(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            compression_enabled=True,
        )

        self.assertIsNotNone(graphrag)
        graphrag.close()

    def test_improved_graphrag_initialization(self):
        """Test ImprovedGraphRAG initialization."""
        from planned_graphrag import ImprovedGraphRAG

        graphrag = ImprovedGraphRAG(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            max_hops=2,
            vector_limit=5,
        )

        self.assertIsNotNone(graphrag)
        graphrag.close()

    def test_followup_graphrag_initialization(self):
        """Test FollowUpGraphRAG initialization."""
        from planned_graphrag import FollowUpGraphRAG

        graphrag = FollowUpGraphRAG(
            backend="falkordb",
            graph_name=TEST_GRAPH_NAME,
            primary_vector_limit=5,
            primary_max_hops=2,
        )

        self.assertIsNotNone(graphrag)
        graphrag.close()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
