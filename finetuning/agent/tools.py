"""
Tool definitions for agent fine-tuning.

Defines 4 tools for GraphRAG agent:
- graph_lookup: Entity/relationship lookup from knowledge graph
- web_search: External search fallback
- cypher_query: Direct Cypher query execution
- entity_resolve: Entity disambiguation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ToolSchema:
    """OpenAI function-calling compatible tool schema."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_qwen3_format(self) -> Dict[str, Any]:
        """Convert to Qwen3 tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# Tool 1: Graph Lookup
GRAPH_LOOKUP = ToolSchema(
    name="graph_lookup",
    description=(
        "Look up an entity and its relationships in the knowledge graph. "
        "Use this to find information about people, organizations, concepts, "
        "events, or locations and how they are connected to other entities."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "The name of the entity to look up (e.g., 'Albert Einstein', 'MIT')"
            },
            "include_relationships": {
                "type": "boolean",
                "description": "Whether to include relationships to other entities",
                "default": True
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum relationship hops from the entity (1-3)",
                "default": 1,
                "minimum": 1,
                "maximum": 3
            },
            "relationship_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to specific relationship types (e.g., ['WORKS_AT', 'AUTHORED']). Empty for all types.",
                "default": []
            }
        },
        "required": ["entity_name"]
    }
)


# Tool 2: Web Search
WEB_SEARCH = ToolSchema(
    name="web_search",
    description=(
        "Search the web for information not found in the knowledge graph. "
        "Use this as a fallback when graph_lookup returns no results or "
        "when the query requires recent/external information."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    }
)


# Tool 3: Cypher Query
CYPHER_QUERY = ToolSchema(
    name="cypher_query",
    description=(
        "Execute a Cypher query directly against the Neo4j knowledge graph. "
        "Use this for complex queries involving aggregations, path finding, "
        "or specific graph patterns that graph_lookup cannot express."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The Cypher query to execute (e.g., 'MATCH (p:Person)-[:WORKS_AT]->(o:Organization) RETURN p.name, o.name LIMIT 10')"
            },
            "params": {
                "type": "object",
                "description": "Query parameters for parameterized queries",
                "default": {}
            }
        },
        "required": ["query"]
    }
)


# Tool 4: Entity Resolve
ENTITY_RESOLVE = ToolSchema(
    name="entity_resolve",
    description=(
        "Resolve an ambiguous entity name to specific entities in the knowledge graph. "
        "Use this when an entity name could refer to multiple entities "
        "(e.g., 'John Smith' could be multiple people, 'Cambridge' could be a city or university)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "The potentially ambiguous entity name"
            },
            "context": {
                "type": "string",
                "description": "Additional context to help disambiguation (e.g., 'physicist', 'in the 1920s', 'American')",
                "default": ""
            },
            "entity_type": {
                "type": "string",
                "description": "Expected entity type to narrow search (e.g., 'Person', 'Organization', 'Location')",
                "default": ""
            },
            "max_candidates": {
                "type": "integer",
                "description": "Maximum number of candidate entities to return",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["entity_name"]
    }
)


# All tools as a list
AGENT_TOOLS: List[ToolSchema] = [
    GRAPH_LOOKUP,
    WEB_SEARCH,
    CYPHER_QUERY,
    ENTITY_RESOLVE,
]

# Tool name to schema mapping
TOOL_MAP: Dict[str, ToolSchema] = {
    tool.name: tool for tool in AGENT_TOOLS
}


def get_tools_for_qwen3() -> List[Dict[str, Any]]:
    """Get all tools in Qwen3 format."""
    return [tool.to_qwen3_format() for tool in AGENT_TOOLS]


def get_tools_for_openai() -> List[Dict[str, Any]]:
    """Get all tools in OpenAI function-calling format."""
    return [tool.to_dict() for tool in AGENT_TOOLS]


def get_tool_by_name(name: str) -> Optional[ToolSchema]:
    """Get a tool schema by name."""
    return TOOL_MAP.get(name)


# Example tool call and response formats for training data
TOOL_CALL_EXAMPLES = {
    "graph_lookup": {
        "call": {
            "name": "graph_lookup",
            "arguments": '{"entity_name": "Albert Einstein", "include_relationships": true, "max_hops": 1}'
        },
        "response": {
            "entity": {
                "name": "Albert Einstein",
                "type": "Person",
                "properties": {
                    "birth_date": "1879-03-14",
                    "death_date": "1955-04-18",
                    "nationality": "German-American"
                }
            },
            "relationships": [
                {"type": "WORKED_AT", "target": "Princeton University"},
                {"type": "DEVELOPED", "target": "Theory of Relativity"},
                {"type": "WON", "target": "Nobel Prize in Physics 1921"}
            ]
        }
    },
    "web_search": {
        "call": {
            "name": "web_search",
            "arguments": '{"query": "latest Nobel Prize in Physics winner", "num_results": 3}'
        },
        "response": {
            "results": [
                {"title": "Nobel Prize 2024", "snippet": "...", "url": "..."},
            ]
        }
    },
    "cypher_query": {
        "call": {
            "name": "cypher_query",
            "arguments": '{"query": "MATCH (p:Person)-[:AUTHORED]->(paper) RETURN p.name, COUNT(paper) as papers ORDER BY papers DESC LIMIT 5"}'
        },
        "response": {
            "records": [
                {"p.name": "Researcher A", "papers": 150},
                {"p.name": "Researcher B", "papers": 120}
            ]
        }
    },
    "entity_resolve": {
        "call": {
            "name": "entity_resolve",
            "arguments": '{"entity_name": "Cambridge", "context": "university in Massachusetts"}'
        },
        "response": {
            "candidates": [
                {"name": "Massachusetts Institute of Technology", "type": "Organization", "score": 0.95},
                {"name": "Harvard University", "type": "Organization", "score": 0.85},
                {"name": "Cambridge, Massachusetts", "type": "Location", "score": 0.60}
            ],
            "best_match": "Massachusetts Institute of Technology"
        }
    }
}
