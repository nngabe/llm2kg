"""
Autonomous research agent for filling knowledge gaps.

Executes research on approved topics with:
- Web search on trusted sources
- Document ingestion into Neo4j
- Entity extraction and linking
- Progress tracking
- Graceful termination
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from .topic_generator import ResearchTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchState(BaseModel):
    """State for the research agent."""

    # Topics
    topics: List[ResearchTopic] = Field(default_factory=list)
    approved_topics: List[int] = Field(default_factory=list)
    current_topic_index: int = 0

    # Time management
    time_limit_minutes: int = 30
    start_time: Optional[str] = None
    should_terminate: bool = False

    # Progress
    documents_added: int = 0
    entities_created: int = 0
    topics_completed: int = 0

    # Current work
    current_query: str = ""
    search_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Error handling
    errors: List[str] = Field(default_factory=list)


class ResearchAgent:
    """
    Autonomous research agent using LangGraph.

    Features:
    - Executes approved research topics
    - Searches trusted web sources
    - Ingests documents into knowledge graph
    - Tracks progress with callbacks
    - Terminates gracefully on time limit
    """

    def __init__(
        self,
        neo4j_loader: Optional[Any] = None,
        web_search: Optional[Any] = None,
        time_limit_minutes: int = 30,
        documents_per_topic: int = 5,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize the research agent.

        Args:
            neo4j_loader: Neo4j loader for database operations.
            web_search: Web search client.
            time_limit_minutes: Maximum research time.
            documents_per_topic: Target documents per topic.
            progress_callback: Callback for progress updates.
        """
        self.time_limit = time_limit_minutes
        self.documents_per_topic = documents_per_topic
        self.progress_callback = progress_callback

        # Initialize components
        self._init_components(neo4j_loader, web_search)

        # Build graph
        self.graph = self._build_graph()

    def _init_components(
        self,
        neo4j_loader: Optional[Any],
        web_search: Optional[Any],
    ):
        """Initialize research components."""
        if neo4j_loader is None:
            from agent_skb import Neo4jAgentLoader, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
            self._neo4j = Neo4jAgentLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        else:
            self._neo4j = neo4j_loader

        if web_search is None:
            try:
                from finetuning.agent.tools import WebSearchTool
                self._web_search = WebSearchTool()
            except ImportError:
                logger.warning("Web search not available")
                self._web_search = None
        else:
            self._web_search = web_search

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(ResearchState)

        # Add nodes
        graph.add_node("check_time", self._check_time_node)
        graph.add_node("get_next_topic", self._get_next_topic_node)
        graph.add_node("search_web", self._search_web_node)
        graph.add_node("ingest_documents", self._ingest_documents_node)
        graph.add_node("update_progress", self._update_progress_node)

        # Add edges
        graph.add_edge(START, "check_time")
        graph.add_conditional_edges(
            "check_time",
            self._should_continue,
            {
                "continue": "get_next_topic",
                "stop": END,
            }
        )
        graph.add_conditional_edges(
            "get_next_topic",
            self._has_topic,
            {
                "has_topic": "search_web",
                "no_topic": END,
            }
        )
        graph.add_edge("search_web", "ingest_documents")
        graph.add_edge("ingest_documents", "update_progress")
        graph.add_edge("update_progress", "check_time")

        return graph.compile()

    def _should_continue(self, state: ResearchState) -> str:
        """Check if research should continue."""
        if state.should_terminate:
            return "stop"

        if state.start_time:
            start = datetime.fromisoformat(state.start_time)
            elapsed = datetime.now() - start
            if elapsed >= timedelta(minutes=state.time_limit_minutes):
                return "stop"

        return "continue"

    def _has_topic(self, state: ResearchState) -> str:
        """Check if there are more topics to process."""
        if state.current_topic_index < len(state.approved_topics):
            return "has_topic"
        return "no_topic"

    def _check_time_node(self, state: ResearchState) -> Dict[str, Any]:
        """Check time and initialize if needed."""
        if not state.start_time:
            return {"start_time": datetime.now().isoformat()}
        return {}

    def _get_next_topic_node(self, state: ResearchState) -> Dict[str, Any]:
        """Get the next topic to research."""
        if state.current_topic_index >= len(state.approved_topics):
            return {}

        topic_number = state.approved_topics[state.current_topic_index]
        topic = next(
            (t for t in state.topics if t.number == topic_number),
            None,
        )

        if not topic:
            return {"current_topic_index": state.current_topic_index + 1}

        # Get first search query
        query = topic.search_queries[0] if topic.search_queries else topic.title

        logger.info(f"Researching topic {topic.number}: {topic.title}")

        return {"current_query": query}

    def _search_web_node(self, state: ResearchState) -> Dict[str, Any]:
        """Execute web search for current topic."""
        if not self._web_search or not state.current_query:
            return {"search_results": []}

        try:
            result = self._web_search.execute(
                query=state.current_query,
                num_results=self.documents_per_topic,
            )

            results = result.get("results", [])
            logger.info(f"Found {len(results)} results for: {state.current_query}")

            return {"search_results": results}

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "search_results": [],
                "errors": state.errors + [str(e)],
            }

    def _ingest_documents_node(self, state: ResearchState) -> Dict[str, Any]:
        """Ingest search results into knowledge graph."""
        documents_added = 0
        entities_created = 0

        for result in state.search_results:
            try:
                # Add document to database
                doc_id = self._neo4j.create_document(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    content=result.get("snippet", ""),
                    domain=result.get("domain", ""),
                    trust_level=result.get("trust_level", "medium"),
                    source_type="web_search",
                )
                documents_added += 1

                # TODO: Extract entities from document and link

            except Exception as e:
                logger.warning(f"Failed to ingest document: {e}")

        logger.info(f"Ingested {documents_added} documents")

        return {
            "documents_added": state.documents_added + documents_added,
            "entities_created": state.entities_created + entities_created,
        }

    def _update_progress_node(self, state: ResearchState) -> Dict[str, Any]:
        """Update progress and move to next topic."""
        new_index = state.current_topic_index + 1

        # Call progress callback if provided
        if self.progress_callback:
            try:
                self.progress_callback({
                    "topics_completed": new_index,
                    "total_topics": len(state.approved_topics),
                    "documents_added": state.documents_added,
                    "entities_created": state.entities_created,
                })
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        return {
            "current_topic_index": new_index,
            "topics_completed": new_index,
            "search_results": [],
            "current_query": "",
        }

    def research(
        self,
        topics: List[ResearchTopic],
        approved_numbers: List[int],
        time_limit_minutes: Optional[int] = None,
    ) -> ResearchState:
        """
        Execute research on approved topics.

        Args:
            topics: All generated topics.
            approved_numbers: Numbers of approved topics.
            time_limit_minutes: Optional override for time limit.

        Returns:
            Final research state with results.
        """
        initial_state = ResearchState(
            topics=topics,
            approved_topics=sorted(approved_numbers),
            time_limit_minutes=time_limit_minutes or self.time_limit,
        )

        try:
            final_state = self.graph.invoke(initial_state)
            return ResearchState(**final_state)
        except Exception as e:
            logger.error(f"Research error: {e}")
            initial_state.errors.append(str(e))
            return initial_state

    async def research_async(
        self,
        topics: List[ResearchTopic],
        approved_numbers: List[int],
        time_limit_minutes: Optional[int] = None,
    ) -> ResearchState:
        """Async version of research."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.research(topics, approved_numbers, time_limit_minutes),
        )

    def terminate(self):
        """Request graceful termination."""
        logger.info("Termination requested")
        # This would need to be implemented with proper state sharing

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_neo4j') and self._neo4j:
            self._neo4j.close()


def run_research_cli():
    """CLI for testing the research agent."""
    from .gap_detector import KnowledgeGapDetector
    from .topic_generator import ResearchTopicGenerator

    print("Research Agent CLI")
    print("=" * 40)

    # Detect gaps
    print("\nDetecting knowledge gaps...")
    detector = KnowledgeGapDetector()
    gaps = detector.identify_gaps(max_gaps=10)
    print(f"Found {len(gaps)} gaps")
    detector.close()

    # Generate topics
    print("\nGenerating research topics...")
    generator = ResearchTopicGenerator()
    topics = generator.generate_topics(gaps, num_topics=5)

    print("\nGenerated Topics:")
    print(generator.format_for_approval(topics))

    # Get approval
    approval = input("\nEnter topic numbers to approve (e.g., 1,2,3): ")
    try:
        approved = [int(x.strip()) for x in approval.split(",")]
    except ValueError:
        print("Invalid input")
        return

    # Run research
    print(f"\nResearching {len(approved)} topics...")
    agent = ResearchAgent(time_limit_minutes=10)

    def progress_cb(data):
        print(f"Progress: {data['topics_completed']}/{data['total_topics']} topics")

    agent.progress_callback = progress_cb
    result = agent.research(topics, approved)

    print("\nResearch Complete!")
    print(f"Documents added: {result.documents_added}")
    print(f"Entities created: {result.entities_created}")
    print(f"Topics completed: {result.topics_completed}")

    agent.close()


if __name__ == "__main__":
    run_research_cli()
