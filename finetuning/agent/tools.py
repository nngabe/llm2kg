"""
Tool definitions for agent fine-tuning.

Defines 4 tools for GraphRAG agent:
- graph_lookup: Entity/relationship lookup from knowledge graph
- web_search: External search fallback (now with Tavily implementation)
- cypher_query: Direct Cypher query execution
- entity_resolve: Entity disambiguation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


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


# =============================================================================
# Tool Executor Classes
# =============================================================================

class WebSearchTool:
    """
    Executor for web_search tool using Tavily API.

    Provides actual web search functionality with trusted source filtering.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        trusted_sources_config: Optional[str] = None,
        filter_trusted_only: bool = True,
    ):
        """
        Initialize the web search tool.

        Args:
            api_key: Tavily API key. If None, uses TAVILY_API_KEY env var.
            trusted_sources_config: Path to trusted sources YAML config.
            filter_trusted_only: Whether to filter results to trusted sources.
        """
        self.filter_trusted_only = filter_trusted_only
        self._client = None
        self._source_manager = None
        self._api_key = api_key
        self._config_path = trusted_sources_config

    def _ensure_initialized(self):
        """Lazy initialization of client and source manager."""
        if self._client is None:
            from web_search import WebSearchClient, TrustedSourceManager
            self._client = WebSearchClient(api_key=self._api_key)
            self._source_manager = TrustedSourceManager(
                config_path=self._config_path,
                auto_load=True,
            )

    def execute(
        self,
        query: str,
        num_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a web search.

        Args:
            query: Search query string.
            num_results: Number of results to return.

        Returns:
            Dictionary with search results.
        """
        self._ensure_initialized()

        try:
            if self.filter_trusted_only:
                trusted_domains = self._source_manager.get_trusted_domains()
                results = self._client.search_trusted_only(
                    query=query,
                    trusted_domains=trusted_domains,
                    num_results=num_results,
                )
            else:
                results = self._client.search(
                    query=query,
                    num_results=num_results,
                )

            return {
                "status": "success",
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.content,
                        "domain": r.domain,
                        "score": r.score,
                        "trust_level": (
                            self._source_manager.get_trust_level(r.url).value
                            if self._source_manager.get_trust_level(r.url)
                            else "unknown"
                        ),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
            }

    def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch full content from a URL.

        Args:
            url: URL to fetch.

        Returns:
            Dictionary with document content.
        """
        self._ensure_initialized()

        try:
            content = self._client.fetch_content(url)
            if content:
                return {
                    "status": "success",
                    "url": url,
                    "title": content.title,
                    "content": content.content,
                    "domain": content.domain,
                    "word_count": content.word_count,
                    "trust_level": (
                        self._source_manager.get_trust_level(url).value
                        if self._source_manager.get_trust_level(url)
                        else "unknown"
                    ),
                }
            else:
                return {
                    "status": "error",
                    "url": url,
                    "error": "Failed to extract content",
                }
        except Exception as e:
            logger.error(f"Content fetch error: {e}")
            return {
                "status": "error",
                "url": url,
                "error": str(e),
            }


class ToolExecutor:
    """
    Unified executor for all agent tools.

    Provides a single interface to execute any defined tool.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        trusted_sources_config: Optional[str] = None,
    ):
        """
        Initialize the tool executor.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            tavily_api_key: Tavily API key for web search.
            trusted_sources_config: Path to trusted sources config.
        """
        import os
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")

        self._web_search = WebSearchTool(
            api_key=tavily_api_key,
            trusted_sources_config=trusted_sources_config,
        )
        self._neo4j_driver = None

    def _ensure_neo4j(self):
        """Lazy initialization of Neo4j driver."""
        if self._neo4j_driver is None:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        if tool_name == "web_search":
            return self._web_search.execute(
                query=arguments.get("query", ""),
                num_results=arguments.get("num_results", 5),
            )
        elif tool_name == "graph_lookup":
            return self._execute_graph_lookup(arguments)
        elif tool_name == "cypher_query":
            return self._execute_cypher_query(arguments)
        elif tool_name == "entity_resolve":
            return self._execute_entity_resolve(arguments)
        else:
            return {
                "status": "error",
                "error": f"Unknown tool: {tool_name}",
            }

    def _execute_graph_lookup(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph_lookup tool."""
        self._ensure_neo4j()

        entity_name = arguments.get("entity_name", "")
        include_rels = arguments.get("include_relationships", True)
        max_hops = arguments.get("max_hops", 1)
        rel_types = arguments.get("relationship_types", [])

        try:
            with self._neo4j_driver.session() as session:
                # Find entity
                entity_query = """
                MATCH (e)
                WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
                RETURN e, labels(e) as labels
                LIMIT 1
                """
                result = session.run(entity_query, name=entity_name)
                record = result.single()

                if not record:
                    return {
                        "status": "not_found",
                        "entity_name": entity_name,
                        "message": f"Entity '{entity_name}' not found in knowledge graph",
                    }

                entity = dict(record["e"])
                entity["labels"] = record["labels"]

                relationships = []
                if include_rels:
                    # Get relationships
                    rel_query = """
                    MATCH (e)-[r]->(t)
                    WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
                    RETURN type(r) as rel_type, r, t, labels(t) as target_labels
                    LIMIT 50
                    """
                    rel_result = session.run(rel_query, name=entity_name)

                    for rec in rel_result:
                        rel_type = rec["rel_type"]
                        if rel_types and rel_type not in rel_types:
                            continue
                        relationships.append({
                            "type": rel_type,
                            "properties": dict(rec["r"]) if rec["r"] else {},
                            "target": dict(rec["t"]),
                            "target_labels": rec["target_labels"],
                        })

                return {
                    "status": "success",
                    "entity": entity,
                    "relationships": relationships,
                }

        except Exception as e:
            logger.error(f"Graph lookup error: {e}")
            return {
                "status": "error",
                "entity_name": entity_name,
                "error": str(e),
            }

    def _execute_cypher_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cypher_query tool."""
        self._ensure_neo4j()

        query = arguments.get("query", "")
        params = arguments.get("params", {})

        try:
            with self._neo4j_driver.session() as session:
                result = session.run(query, **params)
                records = [dict(record) for record in result]

                return {
                    "status": "success",
                    "query": query,
                    "num_records": len(records),
                    "records": records[:100],  # Limit to 100 records
                }

        except Exception as e:
            logger.error(f"Cypher query error: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
            }

    def _execute_entity_resolve(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity_resolve tool."""
        self._ensure_neo4j()

        entity_name = arguments.get("entity_name", "")
        context = arguments.get("context", "")
        entity_type = arguments.get("entity_type", "")
        max_candidates = arguments.get("max_candidates", 5)

        try:
            with self._neo4j_driver.session() as session:
                # Search for candidates
                query = """
                MATCH (e)
                WHERE toLower(e.name) CONTAINS toLower($name)
                   OR toLower(e.id) CONTAINS toLower($name)
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                result = session.run(query, name=entity_name, limit=max_candidates * 2)

                candidates = []
                for record in result:
                    entity = dict(record["e"])
                    entity["labels"] = record["labels"]

                    # Filter by type if specified
                    if entity_type and entity_type not in record["labels"]:
                        continue

                    # Simple scoring based on name match
                    name = entity.get("name", entity.get("id", "")).lower()
                    score = 1.0 if name == entity_name.lower() else 0.5

                    # Boost score if context appears in description
                    desc = entity.get("description", "").lower()
                    if context and context.lower() in desc:
                        score += 0.3

                    candidates.append({
                        "entity": entity,
                        "score": min(score, 1.0),
                    })

                # Sort by score
                candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = candidates[:max_candidates]

                return {
                    "status": "success",
                    "entity_name": entity_name,
                    "num_candidates": len(candidates),
                    "candidates": candidates,
                    "best_match": candidates[0] if candidates else None,
                }

        except Exception as e:
            logger.error(f"Entity resolve error: {e}")
            return {
                "status": "error",
                "entity_name": entity_name,
                "error": str(e),
            }

    def close(self):
        """Close connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
