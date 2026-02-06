#!/usr/bin/env python3
"""
Shared test fixtures and mocks for unit tests.

Provides:
- MockFalkorDBGraph: Mock FalkorDB graph with configurable responses
- MockLLM: Mock ChatOllama with configurable responses
- MockEmbeddings: Mock OllamaEmbeddings returning fake vectors
- Sample data fixtures for entities, chunks, plans, etc.
"""

import sys
sys.path.insert(0, '/app')

from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime


# =============================================================================
# MOCK CLASSES
# =============================================================================

class MockFalkorDBResult:
    """Mock FalkorDB query result."""

    def __init__(self, result_set: List[List[Any]] = None):
        self.result_set = result_set or []


class MockFalkorDBNode:
    """Mock FalkorDB node with properties."""

    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties


class MockFalkorDBGraph:
    """Mock FalkorDB graph with query responses."""

    def __init__(self, responses: Dict[str, List] = None):
        """
        Initialize mock graph.

        Args:
            responses: Dict mapping Cypher query substrings to result sets.
        """
        self.responses = responses or {}
        self.queries = []  # Track all queries made
        self.default_response = []

    def query(self, cypher: str, params: Dict[str, Any] = None) -> MockFalkorDBResult:
        """Execute a mock query and return configured response."""
        self.queries.append((cypher, params))

        # Find matching response by substring
        for pattern, result_set in self.responses.items():
            if pattern in cypher:
                return MockFalkorDBResult(result_set)

        return MockFalkorDBResult(self.default_response)

    def set_response(self, pattern: str, result_set: List):
        """Set response for queries containing pattern."""
        self.responses[pattern] = result_set

    def clear_queries(self):
        """Clear query history."""
        self.queries = []


class MockLLM:
    """Mock ChatOllama with configurable responses."""

    def __init__(self, responses: List[str] = None, default_response: str = "{}"):
        """
        Initialize mock LLM.

        Args:
            responses: List of responses to return in order.
            default_response: Response to return when responses are exhausted.
        """
        self.responses = responses or []
        self.default_response = default_response
        self.call_count = 0
        self.messages = []  # Track all messages received

    def invoke(self, messages) -> 'MockAIMessage':
        """Invoke mock LLM and return response."""
        self.messages.append(messages)

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.default_response

        self.call_count += 1
        return MockAIMessage(content=response)

    def reset(self):
        """Reset call count and message history."""
        self.call_count = 0
        self.messages = []


class MockAIMessage:
    """Mock LangChain AI message."""

    def __init__(self, content: str):
        self.content = content


class MockEmbeddings:
    """Mock OllamaEmbeddings returning fake vectors."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.embed_query_count = 0
        self.embed_documents_count = 0

    def embed_query(self, text: str) -> List[float]:
        """Return fake embedding for query."""
        self.embed_query_count += 1
        # Return deterministic embedding based on text hash
        seed = hash(text) % 1000
        return [0.1 + (i * seed % 100) / 1000 for i in range(self.dimension)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        self.embed_documents_count += len(texts)
        return [self.embed_query(text) for text in texts]


class MockHTTPResponse:
    """Mock HTTP response for requests."""

    def __init__(self, json_data: Dict = None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


class MockSession:
    """Mock requests.Session for web scraping tests."""

    def __init__(self, responses: Dict[str, MockHTTPResponse] = None):
        self.responses = responses or {}
        self.requests_made = []

    def get(self, url: str, **kwargs) -> MockHTTPResponse:
        self.requests_made.append(('GET', url, kwargs))
        return self.responses.get(url, MockHTTPResponse(text="<html></html>"))

    def post(self, url: str, **kwargs) -> MockHTTPResponse:
        self.requests_made.append(('POST', url, kwargs))
        return self.responses.get(url, MockHTTPResponse())


# =============================================================================
# SAMPLE DATA
# =============================================================================

SAMPLE_ENTITY = {
    "name": "GE Vernova",
    "entity_id": "ent_001",
    "qid": "Q123456",
    "description": "American energy company",
    "ontology_type": "ORGANIZATION",
    "labels": ["Entity", "Organization"],
}

SAMPLE_ENTITY_2 = {
    "name": "9HA gas turbine",
    "entity_id": "ent_002",
    "qid": "Q789012",
    "description": "Heavy-duty gas turbine manufactured by GE",
    "ontology_type": "PRODUCT",
    "labels": ["Entity", "Product"],
}

SAMPLE_DOCUMENT_CHUNK = {
    "chunk_id": "chunk_001",
    "content": "The 9HA gas turbine is the world's most efficient gas turbine with >64% combined cycle efficiency.",
    "source_type": "webpage",
    "source_url": "https://www.gevernova.com/gas-power",
    "chunk_type": "leaf",
}

SAMPLE_EPISODE = {
    "episode_id": "ep_001",
    "name": "GE Vernova Product Page",
    "content": "Information about 9HA gas turbine specifications.",
    "reference_time": "2024-01-15T10:30:00Z",
    "source_url": "https://www.gevernova.com/gas-power",
    "source_type": "webpage",
    "confidence": 0.95,
}

SAMPLE_COMMUNITY = {
    "community_id": "comm_001",
    "name": "Gas Power Products",
    "members": ["9HA gas turbine", "7HA gas turbine", "LM6000 gas turbine"],
}

SAMPLE_RELATIONSHIP = {
    "type": "MANUFACTURES",
    "properties": {"confidence": 0.9},
    "other_entity": SAMPLE_ENTITY_2,
    "direction": "outgoing",
}

SAMPLE_RETRIEVAL_PLAN = {
    "entity_targets": ["GE Vernova", "9HA gas turbine"],
    "relationship_queries": ["MANUFACTURES", "PART_OF"],
    "information_needs": ["efficiency", "power output"],
    "reasoning": "Need to find product specifications for 9HA turbine.",
}

SAMPLE_THINK_RESPONSE = """{
    "thought": "I need to look up the 9HA turbine specifications in the knowledge graph.",
    "ready_to_answer": false,
    "action": {
        "tool_name": "graph_lookup",
        "arguments": {"entity_name": "9HA gas turbine"}
    }
}"""

SAMPLE_THINK_RESPONSE_READY = """{
    "thought": "I now have enough information to answer the question about 9HA turbine efficiency.",
    "ready_to_answer": true,
    "action": {
        "tool_name": "none",
        "arguments": {}
    }
}"""

SAMPLE_ANSWER_RESPONSE = """{
    "answer": "The 9HA gas turbine has a combined cycle efficiency of over 64%, making it the world's most efficient gas turbine. It is manufactured by GE Vernova.",
    "citations": [
        {
            "source_type": "graph",
            "source_id": "9HA gas turbine",
            "source_title": "9HA Gas Turbine",
            "excerpt": "combined cycle efficiency >64%"
        },
        {
            "source_type": "document",
            "source_id": "https://www.gevernova.com/gas-power",
            "source_title": "GE Vernova Gas Power",
            "excerpt": "world's most efficient gas turbine"
        }
    ]
}"""

SAMPLE_WIKIDATA_ENTITY = {
    "qid": "Q1101",
    "name": "Technology",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Technology",
    "description": "application of knowledge for practical goals",
}

SAMPLE_SEEDS = [
    {"qid": "Q15635738", "name": "GE Vernova"},
    {"qid": "Q2624", "name": "power plant"},
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_graph_result(entities: List[Dict], include_node_wrapper: bool = True) -> List[List]:
    """Create mock FalkorDB result set from entity dicts."""
    result_set = []
    for entity in entities:
        if include_node_wrapper:
            node = MockFalkorDBNode(entity)
            result_set.append([node, entity.get("labels", [])])
        else:
            result_set.append([entity])
    return result_set


def create_mock_entity_result(entity: Dict, relationships: List[Dict] = None) -> Dict:
    """Create a mock entity lookup result."""
    return {
        "found": True,
        "entity": entity,
        "relationships": relationships or [],
        "episode_count": 3,
        "community": {"name": "Gas Power Products", "id": "comm_001"},
    }


def create_mock_llm_with_responses(think_responses: List[str], answer_response: str) -> MockLLM:
    """Create a MockLLM configured for ReAct agent testing."""
    responses = think_responses + [answer_response]
    return MockLLM(responses=responses)
