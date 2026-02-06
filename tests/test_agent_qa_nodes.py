#!/usr/bin/env python3
"""
Unit tests for agent_qa.py nodes and tools.

Tests cover all ReAct agent nodes:
- _plan_retrieval_node: CLaRa-style retrieval planning
- _retrieve_planned_node: Execute retrieval plan
- _think_node: Generate next thought and action
- _execute_action_node: Execute tool calls
- _compress_observation_node: Compress tool observations
- _synthesize_node: Generate final answer

Also tests:
- Database loaders (Neo4jQALoader, FalkorDBQALoader)
- Tool execution (graph_lookup, wiki_search, web_search, etc.)
- Agent integration (full answer flow)
"""

import sys
sys.path.insert(0, '/app')

import json
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, Any, List

# Import test fixtures
from conftest import (
    MockFalkorDBGraph,
    MockFalkorDBResult,
    MockFalkorDBNode,
    MockLLM,
    MockAIMessage,
    MockEmbeddings,
    MockHTTPResponse,
    SAMPLE_ENTITY,
    SAMPLE_ENTITY_2,
    SAMPLE_DOCUMENT_CHUNK,
    SAMPLE_EPISODE,
    SAMPLE_COMMUNITY,
    SAMPLE_RELATIONSHIP,
    SAMPLE_RETRIEVAL_PLAN,
    SAMPLE_THINK_RESPONSE,
    SAMPLE_THINK_RESPONSE_READY,
    SAMPLE_ANSWER_RESPONSE,
    create_mock_entity_result,
    create_mock_llm_with_responses,
)

# Import modules under test
from agent_qa import (
    QAAgentState,
    ToolCall,
    ToolResult,
    Citation,
    ThoughtStep,
    QAResponse,
    ContextItem,
    ReActQAAgent,
    Neo4jQALoader,
    FalkorDBQALoader,
    create_qa_loader,
)

from planned_graphrag import RetrievalPlan


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestDataModels(unittest.TestCase):
    """Tests for agent_qa data models."""

    def test_tool_call_creation(self):
        """Test ToolCall model creation."""
        call = ToolCall(
            tool_name="graph_lookup",
            arguments={"entity_name": "9HA gas turbine"},
        )
        self.assertEqual(call.tool_name, "graph_lookup")
        self.assertEqual(call.arguments["entity_name"], "9HA gas turbine")
        self.assertIsNotNone(call.timestamp)

    def test_citation_creation(self):
        """Test Citation model creation with valid source types."""
        citation = Citation(
            source_type="graph",
            source_id="9HA gas turbine",
            source_title="9HA Gas Turbine",
            excerpt="combined cycle efficiency >64%",
        )
        self.assertEqual(citation.source_type, "graph")
        self.assertEqual(citation.source_id, "9HA gas turbine")

    def test_thought_step_creation(self):
        """Test ThoughtStep model creation."""
        step = ThoughtStep(
            thought="I need to look up the 9HA turbine",
            action=ToolCall(tool_name="graph_lookup", arguments={"entity_name": "9HA"}),
            observation="Entity found with 64% efficiency",
        )
        self.assertEqual(step.thought, "I need to look up the 9HA turbine")
        self.assertIsNotNone(step.action)

    def test_qa_response_creation(self):
        """Test QAResponse model creation."""
        response = QAResponse(
            question="What is the efficiency of the 9HA turbine?",
            answer="The 9HA has >64% combined cycle efficiency.",
            citations=[Citation(source_type="graph", source_id="9HA")],
            confidence=0.85,
        )
        self.assertEqual(response.question, "What is the efficiency of the 9HA turbine?")
        self.assertEqual(len(response.citations), 1)

    def test_context_item_creation(self):
        """Test ContextItem model creation."""
        item = ContextItem(
            source_type="entity",
            content="[Entity: 9HA gas turbine]",
            source_id="9HA gas turbine",
            relevance_score=0.95,
        )
        self.assertEqual(item.source_type, "entity")
        self.assertEqual(item.relevance_score, 0.95)

    def test_qa_agent_state_defaults(self):
        """Test QAAgentState default values."""
        state = QAAgentState(question="Test question")
        self.assertEqual(state.question, "Test question")
        self.assertEqual(state.iteration_count, 0)
        self.assertEqual(state.max_iterations, 5)
        self.assertFalse(state.ready_to_answer)
        self.assertIsNone(state.error)


# =============================================================================
# PLAN RETRIEVAL NODE TESTS
# =============================================================================

class TestPlanRetrievalNode(unittest.TestCase):
    """Tests for _plan_retrieval_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLM(responses=[
            json.dumps(SAMPLE_RETRIEVAL_PLAN)
        ])
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_generates_valid_plan(self, mock_planned_rag):
        """Test that valid retrieval plan is generated."""
        mock_planned_rag.return_value = Mock()

        agent = ReActQAAgent(
            llm=self.mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="What is the efficiency of the 9HA turbine?")
        result = agent._plan_retrieval_node(state)

        self.assertIn("retrieval_plan", result)
        plan = result["retrieval_plan"]
        self.assertIsNotNone(plan)

    @patch('agent_qa.PlannedGraphRAG')
    def test_empty_plan_on_error(self, mock_planned_rag):
        """Test that empty plan is returned on LLM failure."""
        mock_planned_rag.return_value = Mock()

        # LLM raises exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test question")
        result = agent._plan_retrieval_node(state)

        self.assertIn("retrieval_plan", result)
        plan = result["retrieval_plan"]
        # Empty plan should be returned
        self.assertEqual(len(plan.entity_targets), 0)


# =============================================================================
# RETRIEVE PLANNED NODE TESTS
# =============================================================================

class TestRetrievePlannedNode(unittest.TestCase):
    """Tests for _retrieve_planned_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_fallback_on_empty_plan(self, mock_planned_rag):
        """Test fallback to basic retrieval when plan is empty."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=["{}"])

        # Mock db_loader to return empty lists
        self.mock_db_loader.vector_search.return_value = []
        self.mock_db_loader.search_documents.return_value = []

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        # Empty retrieval plan
        state = QAAgentState(
            question="Test question",
            retrieval_plan=RetrievalPlan(),
        )
        result = agent._retrieve_planned_node(state)

        # Should return context (possibly empty)
        self.assertIn("context", result)

    @patch('agent_qa.ImprovedGraphRAG')
    def test_improved_graphrag_retrieval(self, mock_improved_rag):
        """Test using ImprovedGraphRAG when enabled."""
        mock_improved_rag_instance = Mock()
        mock_improved_rag_instance.retrieve_with_keywords.return_value = Mock(
            entities=[],
            raw_text="",
            compressed_text="",
        )
        mock_improved_rag.return_value = mock_improved_rag_instance

        mock_llm = MockLLM(responses=[json.dumps(SAMPLE_RETRIEVAL_PLAN)])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
            use_improved_retrieval=True,
        )

        state = QAAgentState(
            question="Test question",
            retrieval_plan=RetrievalPlan(entity_targets=["9HA"]),
        )
        result = agent._retrieve_planned_node(state)

        self.assertIn("context", result)


# =============================================================================
# THINK NODE TESTS
# =============================================================================

class TestThinkNode(unittest.TestCase):
    """Tests for _think_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_generates_thought(self, mock_planned_rag):
        """Test that thought is generated."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_THINK_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="What is the efficiency of the 9HA turbine?",
            context_formatted="Sample context",
        )
        result = agent._think_node(state)

        self.assertIn("current_thought", result)
        self.assertIn("I need to look up the 9HA turbine", result["current_thought"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_ready_to_answer_true(self, mock_planned_rag):
        """Test that ready_to_answer is set when sufficient context."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_THINK_RESPONSE_READY])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test question",
            context_formatted="Sufficient context here",
        )
        result = agent._think_node(state)

        self.assertTrue(result["ready_to_answer"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_action_tool_call(self, mock_planned_rag):
        """Test that ToolCall is returned for next action."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_THINK_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test question")
        result = agent._think_node(state)

        self.assertIn("pending_action", result)
        action = result["pending_action"]
        self.assertEqual(action.tool_name, "graph_lookup")
        self.assertEqual(action.arguments["entity_name"], "9HA gas turbine")

    @patch('agent_qa.PlannedGraphRAG')
    def test_json_repair(self, mock_planned_rag):
        """Test handling malformed JSON with json_repair."""
        mock_planned_rag.return_value = Mock()

        # Malformed JSON (missing closing brace)
        malformed_response = '{"thought": "test", "ready_to_answer": true, "action": {"tool_name": "none"}'

        mock_llm = MockLLM(responses=[malformed_response])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test question")
        result = agent._think_node(state)

        # Should handle gracefully (either repair or return error)
        self.assertIn("current_thought", result)

    @patch('agent_qa.PlannedGraphRAG')
    def test_empty_response_handling(self, mock_planned_rag):
        """Test handling empty LLM response."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[""])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test question")
        result = agent._think_node(state)

        # Should set ready_to_answer with error
        self.assertTrue(result.get("ready_to_answer", False))
        self.assertIn("error", result)

    @patch('agent_qa.PlannedGraphRAG')
    def test_iteration_count_increment(self, mock_planned_rag):
        """Test that iteration_count is incremented."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_THINK_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test", iteration_count=2)
        result = agent._think_node(state)

        self.assertEqual(result["iteration_count"], 3)


# =============================================================================
# EXECUTE ACTION NODE TESTS
# =============================================================================

class TestExecuteActionNode(unittest.TestCase):
    """Tests for _execute_action_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_graph_lookup_found(self, mock_planned_rag):
        """Test graph_lookup when entity is found."""
        mock_planned_rag.return_value = Mock()

        # Mock entity lookup
        self.mock_db_loader.get_entity_with_relationships.return_value = {
            "found": True,
            "entity": SAMPLE_ENTITY_2,
            "relationships": [SAMPLE_RELATIONSHIP],
        }
        self.mock_db_loader.get_entity_episodes = Mock(return_value=[SAMPLE_EPISODE])
        self.mock_db_loader.get_community_members = Mock(return_value=[])

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="graph_lookup",
                arguments={"entity_name": "9HA gas turbine"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("last_observation", result)
        self.assertIn("9HA gas turbine", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_graph_lookup_not_found(self, mock_planned_rag):
        """Test graph_lookup when entity is not found."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.get_entity_with_relationships.return_value = {
            "found": False,
            "entity_name": "Unknown Entity",
        }

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="graph_lookup",
                arguments={"entity_name": "Unknown Entity"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("not found", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_graph_lookup_with_episodes(self, mock_planned_rag):
        """Test graph_lookup includes Zep episode data."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.get_entity_with_relationships.return_value = {
            "found": True,
            "entity": SAMPLE_ENTITY_2,
            "relationships": [],
        }
        self.mock_db_loader.get_entity_episodes = Mock(return_value=[SAMPLE_EPISODE])
        self.mock_db_loader.get_community_members = Mock(return_value=[])

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="graph_lookup",
                arguments={"entity_name": "9HA gas turbine"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        # Episodes should be fetched
        self.mock_db_loader.get_entity_episodes.assert_called()

    @patch('agent_qa.PlannedGraphRAG')
    def test_wiki_search_disabled(self, mock_planned_rag):
        """Test wiki_search returns error when disabled."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
            wiki_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="wiki_search",
                arguments={"query": "gas turbine"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("not enabled", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_web_search_disabled(self, mock_planned_rag):
        """Test web_search returns error when disabled."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="web_search",
                arguments={"query": "GE Vernova"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("not enabled", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_cypher_query_success(self, mock_planned_rag):
        """Test cypher_query execution."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.run_cypher.return_value = [
            {"name": "9HA", "efficiency": "64%"},
        ]

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="cypher_query",
                arguments={"query": "MATCH (e:Entity) RETURN e.name, e.efficiency LIMIT 1"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("Query returned", result["last_observation"])
        self.mock_db_loader.run_cypher.assert_called()

    @patch('agent_qa.PlannedGraphRAG')
    def test_entity_resolve_candidates(self, mock_planned_rag):
        """Test entity_resolve returns disambiguation candidates."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.vector_search.return_value = [
            {"name": "9HA gas turbine", "ontology_type": "PRODUCT", "description": "Heavy-duty turbine", "score": 0.95},
            {"name": "9F gas turbine", "ontology_type": "PRODUCT", "description": "Another turbine", "score": 0.80},
        ]

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="entity_resolve",
                arguments={"entity_name": "9H turbine", "context": "gas power"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("Found 2 candidates", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_entity_timeline_success(self, mock_planned_rag):
        """Test entity_timeline returns temporal history."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.get_entity_timeline = Mock(return_value=[
            {"observed_at": "2024-01-15", "source": "webpage", "url": "https://example.com", "excerpt": "Info about 9HA"},
        ])

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="entity_timeline",
                arguments={"entity_name": "9HA gas turbine"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        self.assertIn("Temporal history", result["last_observation"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_pass_error_to_agent(self, mock_planned_rag):
        """Test tool error becomes observation when configured."""
        mock_planned_rag.return_value = Mock()

        self.mock_db_loader.run_cypher.side_effect = Exception("Cypher syntax error")

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
            pass_tool_errors_to_agent=True,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(
                tool_name="cypher_query",
                arguments={"query": "INVALID CYPHER"},
            ),
            context=[],
        )
        result = agent._execute_action_node(state)

        # Error should be passed as observation
        self.assertIn("Error", result["last_observation"])


# =============================================================================
# COMPRESS OBSERVATION NODE TESTS
# =============================================================================

class TestCompressObservationNode(unittest.TestCase):
    """Tests for _compress_observation_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_compresses_long_observation(self, mock_planned_rag):
        """Test that long observations are compressed."""
        mock_graphrag = Mock()
        mock_graphrag.compress_observation.return_value = "Compressed content"
        mock_planned_rag.return_value = mock_graphrag

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
            compression_enabled=True,
        )

        long_observation = "A" * 1000  # > 500 chars

        state = QAAgentState(
            question="Test",
            last_observation=long_observation,
            current_thought="Testing",
            pending_action=ToolCall(tool_name="graph_lookup", arguments={}),
            thought_history=[],
        )
        result = agent._compress_observation_node(state)

        self.assertIn("thought_history", result)
        self.assertEqual(len(result["thought_history"]), 1)

    @patch('agent_qa.PlannedGraphRAG')
    def test_short_observation_unchanged(self, mock_planned_rag):
        """Test that short observations pass through unchanged."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM()

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        short_observation = "Short observation"  # < 500 chars

        state = QAAgentState(
            question="Test",
            last_observation=short_observation,
            current_thought="Testing",
            pending_action=ToolCall(tool_name="graph_lookup", arguments={}),
            thought_history=[],
        )
        result = agent._compress_observation_node(state)

        # Observation should be unchanged
        self.assertEqual(result["last_observation"], short_observation)


# =============================================================================
# SYNTHESIZE NODE TESTS
# =============================================================================

class TestSynthesizeNode(unittest.TestCase):
    """Tests for _synthesize_node."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_generates_answer(self, mock_planned_rag):
        """Test that final answer is generated."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_ANSWER_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="What is the efficiency of the 9HA turbine?",
            context_formatted="9HA has >64% efficiency",
            thought_history=[],
        )
        result = agent._synthesize_node(state)

        self.assertIn("final_answer", result)
        self.assertIn("64%", result["final_answer"])

    @patch('agent_qa.PlannedGraphRAG')
    def test_parses_citations(self, mock_planned_rag):
        """Test that citations are extracted."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_ANSWER_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test question",
            context_formatted="Test context",
            thought_history=[],
        )
        result = agent._synthesize_node(state)

        self.assertIn("citations", result)
        self.assertEqual(len(result["citations"]), 2)

    @patch('agent_qa.PlannedGraphRAG')
    def test_normalizes_source_type(self, mock_planned_rag):
        """Test that source_type variants are mapped to valid enum."""
        mock_planned_rag.return_value = Mock()

        # Response with non-standard source_type
        response_with_variants = """{
            "answer": "Test answer",
            "citations": [
                {"source_type": "graph_query", "source_id": "test1"},
                {"source_type": "knowledge_graph", "source_id": "test2"},
                {"source_type": "invalid_type", "source_id": "test3"}
            ]
        }"""

        mock_llm = MockLLM(responses=[response_with_variants])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            context_formatted="",
            thought_history=[],
        )
        result = agent._synthesize_node(state)

        # All should be normalized to "graph"
        for citation in result["citations"]:
            self.assertEqual(citation.source_type, "graph")

    @patch('agent_qa.PlannedGraphRAG')
    def test_skip_uncertainty(self, mock_planned_rag):
        """Test that uncertainty is skipped when skip_uncertainty=True."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_ANSWER_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            context_formatted="",
            thought_history=[],
        )
        result = agent._synthesize_node(state)

        # Uncertainty should be None
        self.assertIsNone(result.get("uncertainty_scores"))

    @patch('agent_qa.PlannedGraphRAG')
    def test_handles_list_response(self, mock_planned_rag):
        """Test handling when LLM returns array instead of dict."""
        mock_planned_rag.return_value = Mock()

        # LLM returns array
        array_response = '[{"answer": "Array answer", "citations": []}]'

        mock_llm = MockLLM(responses=[array_response])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            context_formatted="",
            thought_history=[],
        )
        result = agent._synthesize_node(state)

        self.assertIn("final_answer", result)
        self.assertEqual(result["final_answer"], "Array answer")

    @patch('agent_qa.PlannedGraphRAG')
    def test_external_info_prefix(self, mock_planned_rag):
        """Test that prefix is added when external_info_used=True."""
        mock_planned_rag.return_value = Mock()

        mock_llm = MockLLM(responses=[SAMPLE_ANSWER_RESPONSE])

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            context_formatted="",
            thought_history=[],
            external_info_used=True,
        )
        result = agent._synthesize_node(state)

        # Check for the web search notice
        self.assertIn("web search", result["final_answer"].lower())


# =============================================================================
# SHOULD CONTINUE TESTS
# =============================================================================

class TestShouldContinue(unittest.TestCase):
    """Tests for _should_continue routing."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_returns_end_on_error(self, mock_planned_rag):
        """Test that error state returns 'end'."""
        mock_planned_rag.return_value = Mock()

        agent = ReActQAAgent(
            llm=MockLLM(),
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test", error="Some error")
        result = agent._should_continue(state)

        self.assertEqual(result, "end")

    @patch('agent_qa.PlannedGraphRAG')
    def test_returns_answer_when_ready(self, mock_planned_rag):
        """Test that ready_to_answer returns 'answer'."""
        mock_planned_rag.return_value = Mock()

        agent = ReActQAAgent(
            llm=MockLLM(),
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test", ready_to_answer=True)
        result = agent._should_continue(state)

        self.assertEqual(result, "answer")

    @patch('agent_qa.PlannedGraphRAG')
    def test_returns_answer_on_max_iterations(self, mock_planned_rag):
        """Test that max iterations returns 'answer'."""
        mock_planned_rag.return_value = Mock()

        agent = ReActQAAgent(
            llm=MockLLM(),
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(question="Test", iteration_count=5, max_iterations=5)
        result = agent._should_continue(state)

        self.assertEqual(result, "answer")

    @patch('agent_qa.PlannedGraphRAG')
    def test_returns_execute_with_pending(self, mock_planned_rag):
        """Test that pending_action returns 'execute'."""
        mock_planned_rag.return_value = Mock()

        agent = ReActQAAgent(
            llm=MockLLM(),
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        state = QAAgentState(
            question="Test",
            pending_action=ToolCall(tool_name="graph_lookup", arguments={}),
        )
        result = agent._should_continue(state)

        self.assertEqual(result, "execute")


# =============================================================================
# DATABASE LOADER TESTS
# =============================================================================

class TestFalkorDBQALoader(unittest.TestCase):
    """Tests for FalkorDBQALoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_graph = MockFalkorDBGraph()

    def test_loader_class_exists(self):
        """Test FalkorDBQALoader class exists."""
        self.assertIsNotNone(FalkorDBQALoader)

    def test_loader_methods_exist(self):
        """Test FalkorDBQALoader has required methods."""
        self.assertTrue(hasattr(FalkorDBQALoader, 'vector_search'))
        self.assertTrue(hasattr(FalkorDBQALoader, 'get_entity_with_relationships'))
        self.assertTrue(hasattr(FalkorDBQALoader, 'search_documents'))
        self.assertTrue(hasattr(FalkorDBQALoader, 'get_entity_episodes'))
        self.assertTrue(hasattr(FalkorDBQALoader, 'get_community_members'))
        self.assertTrue(hasattr(FalkorDBQALoader, 'run_cypher'))

    def test_mock_graph_query(self):
        """Test mock graph query returns expected results."""
        # Mock entity lookup
        self.mock_graph.set_response("MATCH (e:Entity)", [
            [MockFalkorDBNode(SAMPLE_ENTITY), ["Entity"]],
        ])

        result = self.mock_graph.query("MATCH (e:Entity) RETURN e")
        self.assertEqual(len(result.result_set), 1)

    def test_mock_graph_episodes_query(self):
        """Test mock graph episode query."""
        self.mock_graph.set_response("MATCH (ep:Episode)-[r:CONTAINS]->(e:Entity)", [
            ["ep_001", "Test Episode", "Content", "2024-01-15", "https://example.com", "webpage", 0.95],
        ])

        result = self.mock_graph.query("MATCH (ep:Episode)-[r:CONTAINS]->(e:Entity) RETURN ep")
        self.assertEqual(len(result.result_set), 1)

    def test_mock_graph_community_query(self):
        """Test mock graph community query."""
        self.mock_graph.set_response("MATCH (e1:Entity)-[:BELONGS_TO]->(c:Community)", [
            ["ent_002", "7HA gas turbine", "PRODUCT", "Another turbine", "Gas Power Products", "comm_001"],
        ])

        result = self.mock_graph.query("MATCH (e1:Entity)-[:BELONGS_TO]->(c:Community) RETURN e1, c")
        self.assertEqual(len(result.result_set), 1)


class TestCreateQALoader(unittest.TestCase):
    """Tests for create_qa_loader factory function."""

    @patch('agent_qa.FalkorDBQALoader')
    def test_creates_falkordb_loader(self, mock_loader_cls):
        """Test creating FalkorDB loader."""
        mock_loader_cls.return_value = Mock()

        loader = create_qa_loader(backend="falkordb", graph_name="test_graph")

        mock_loader_cls.assert_called_once_with(graph_name="test_graph")

    @patch('agent_qa.Neo4jQALoader')
    def test_creates_neo4j_loader(self, mock_loader_cls):
        """Test creating Neo4j loader."""
        mock_loader_cls.return_value = Mock()

        loader = create_qa_loader(backend="neo4j")

        mock_loader_cls.assert_called_once()


# =============================================================================
# AGENT INTEGRATION TESTS
# =============================================================================

class TestAgentIntegration(unittest.TestCase):
    """Integration tests for full ReActQAAgent flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_loader = Mock()
        self.mock_db_loader.embedding_model = MockEmbeddings()

    @patch('agent_qa.PlannedGraphRAG')
    def test_simple_question_from_kg(self, mock_planned_rag):
        """Test answering a simple question from KG only."""
        mock_graphrag = Mock()
        mock_graphrag.retrieve_with_plan.return_value = Mock(
            entities=[],
            relationships=[],
            raw_text="",
            compressed_text="9HA has 64% efficiency",
        )
        mock_planned_rag.return_value = mock_graphrag

        # Mock LLM responses for full flow
        mock_llm = MockLLM(responses=[
            json.dumps(SAMPLE_RETRIEVAL_PLAN),  # Plan
            SAMPLE_THINK_RESPONSE_READY,  # Think (ready to answer)
            SAMPLE_ANSWER_RESPONSE,  # Answer
        ])

        self.mock_db_loader.vector_search.return_value = [SAMPLE_ENTITY_2]
        self.mock_db_loader.search_documents.return_value = []

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        response = agent.answer_question("What is the efficiency of the 9HA turbine?")

        self.assertIsInstance(response, QAResponse)
        self.assertIn("64%", response.answer)

    @patch('agent_qa.PlannedGraphRAG')
    def test_max_iterations_limit(self, mock_planned_rag):
        """Test that agent stops at max_iterations."""
        mock_graphrag = Mock()
        mock_graphrag.retrieve_with_plan.return_value = Mock(
            entities=[],
            relationships=[],
            raw_text="",
            compressed_text="",
        )
        mock_planned_rag.return_value = mock_graphrag

        # LLM always requests more actions
        mock_llm = MockLLM(responses=[
            json.dumps(SAMPLE_RETRIEVAL_PLAN),  # Plan
            SAMPLE_THINK_RESPONSE,  # Think (not ready)
            SAMPLE_THINK_RESPONSE,  # Think again
            SAMPLE_THINK_RESPONSE,  # Think again
            SAMPLE_THINK_RESPONSE,  # Think again
            SAMPLE_THINK_RESPONSE,  # Think again (max iterations)
            SAMPLE_ANSWER_RESPONSE,  # Forced answer
        ])

        self.mock_db_loader.get_entity_with_relationships.return_value = {"found": False}
        self.mock_db_loader.vector_search.return_value = []
        self.mock_db_loader.search_documents.return_value = []

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
        )

        response = agent.answer_question("Test question")

        # Should still produce an answer
        self.assertIsInstance(response, QAResponse)

    @patch('agent_qa.PlannedGraphRAG')
    def test_error_recovery(self, mock_planned_rag):
        """Test agent recovers from errors."""
        mock_graphrag = Mock()
        mock_graphrag.retrieve_with_plan.return_value = Mock(
            entities=[],
            relationships=[],
            raw_text="",
            compressed_text="",
        )
        mock_planned_rag.return_value = mock_graphrag

        mock_llm = MockLLM(responses=[
            json.dumps(SAMPLE_RETRIEVAL_PLAN),  # Plan
            SAMPLE_THINK_RESPONSE,  # Think
            SAMPLE_THINK_RESPONSE_READY,  # Think (ready)
            SAMPLE_ANSWER_RESPONSE,  # Answer
        ])

        # First lookup fails
        self.mock_db_loader.get_entity_with_relationships.side_effect = [
            Exception("DB error"),
            {"found": True, "entity": SAMPLE_ENTITY},
        ]
        self.mock_db_loader.vector_search.return_value = []
        self.mock_db_loader.search_documents.return_value = []

        agent = ReActQAAgent(
            llm=mock_llm,
            db_loader=self.mock_db_loader,
            skip_uncertainty=True,
            web_search_enabled=False,
            pass_tool_errors_to_agent=True,
        )

        response = agent.answer_question("Test question")

        # Should still produce an answer
        self.assertIsInstance(response, QAResponse)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
