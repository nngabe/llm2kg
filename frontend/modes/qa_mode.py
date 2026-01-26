"""
Q&A Mode Handler for Chainlit.

Handles question answering using ReAct agent with:
- Hybrid GraphRAG + document retrieval
- Web search fallback
- Citation tracking
- ReAct step visualization
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any

# Add parent directory to path for agent_qa import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chainlit as cl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAModeHandler:
    """
    Handler for Q&A mode in Chainlit.

    Features:
    - ReAct step visualization (Thought-Action-Observation)
    - Citation display with source indicators
    - External source badge when web search used
    - Interactive graph visualization
    """

    def __init__(self):
        """Initialize Q&A mode handler."""
        self._agent = None
        self._visualizer = None

    async def initialize(self) -> bool:
        """
        Initialize the Q&A agent.

        Returns:
            True if initialization successful.
        """
        try:
            from agent_qa import AsyncReActQAAgent
            # Match agent_qa.py CLI defaults for consistent behavior
            self._agent = AsyncReActQAAgent(
                web_search_enabled=True,
                auto_add_documents=True,
                use_retrieval_planning=True,
                compression_enabled=True,
                wiki_search_enabled=True,
                wiki_max_results=2,
                skip_uncertainty=True,  # Skip for faster UI response
                parse_response_max_retries=2,
                tool_call_max_retries=1,
            )
            logger.info("Q&A Agent initialized successfully")

            from frontend.components.graph_visualizer import GraphVisualizer
            self._visualizer = GraphVisualizer(height="350px")

            return True
        except ImportError as e:
            logger.error(f"Failed to import Q&A agent modules: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Q&A mode: {e}", exc_info=True)
            return False

    async def handle_message(self, message: cl.Message) -> None:
        """
        Handle a user message in Q&A mode.

        Args:
            message: The user's message.
        """
        query = message.content

        if not self._agent:
            await cl.Message(
                content="Q&A agent not initialized. Please refresh.",
                author="System",
            ).send()
            return

        # Create parent step for the full ReAct loop
        async with cl.Step(name="Q&A Processing", type="run") as parent_step:
            parent_step.input = query

            # Show thinking indicator
            thinking_msg = await cl.Message(
                content="Thinking...",
                author="Assistant",
            ).send()

            try:
                # Run the ReAct agent
                response = await self._agent.answer_question(query)

                # Display reasoning steps
                await self._display_reasoning_steps(
                    response.reasoning_steps,
                    parent_step.id,
                )

                # Display citations
                if response.citations:
                    await self._display_citations(
                        response.citations,
                        parent_step.id,
                    )

                # Update the answer message
                answer_content = self._format_answer(response)
                thinking_msg.content = answer_content
                await thinking_msg.update()

                # Add confidence indicator
                confidence_emoji = self._get_confidence_emoji(response.confidence)
                await cl.Message(
                    content=f"Confidence: {confidence_emoji} {response.confidence:.0%}",
                    author="System",
                    parent_id=parent_step.id,
                ).send()

                parent_step.output = "Question answered successfully"

            except Exception as e:
                logger.error(f"Q&A error: {e}")
                thinking_msg.content = f"I encountered an error: {str(e)}"
                await thinking_msg.update()
                parent_step.output = f"Error: {str(e)}"

    async def _display_reasoning_steps(
        self,
        steps: List,
        parent_id: str,
    ) -> None:
        """Display ReAct reasoning steps."""
        if not steps:
            return

        async with cl.Step(
            name="Reasoning Process",
            type="tool",
        ) as reasoning_step:
            reasoning_step.input = f"{len(steps)} reasoning steps"

            content_parts = []
            for i, step in enumerate(steps, 1):
                parts = [f"**Step {i}:**"]

                if hasattr(step, 'thought') and step.thought:
                    parts.append(f"- Thought: {step.thought[:200]}...")

                if hasattr(step, 'action') and step.action:
                    action = step.action
                    parts.append(f"- Action: `{action.tool_name}`")

                if hasattr(step, 'observation') and step.observation:
                    obs_preview = step.observation[:150] + "..." if len(step.observation) > 150 else step.observation
                    parts.append(f"- Observation: {obs_preview}")

                content_parts.append("\n".join(parts))

            await cl.Message(
                content="\n\n".join(content_parts),
                author="System",
                parent_id=parent_id,
            ).send()

            reasoning_step.output = f"Displayed {len(steps)} reasoning steps"

    async def _display_citations(
        self,
        citations: List,
        parent_id: str,
    ) -> None:
        """Display citations with source indicators."""
        if not citations:
            return

        async with cl.Step(name="Sources", type="retrieval") as cite_step:
            cite_step.input = f"{len(citations)} sources"

            content_parts = ["**Sources:**"]

            for cit in citations:
                source_type = getattr(cit, 'source_type', 'unknown')
                source_id = getattr(cit, 'source_id', 'Unknown')
                trust_level = getattr(cit, 'trust_level', 'unknown')

                # Emoji based on source type
                if source_type == "graph":
                    emoji = "🔗"
                    type_label = "Knowledge Graph"
                elif source_type == "document":
                    emoji = "📄"
                    type_label = "Document"
                elif source_type == "web_search":
                    emoji = "🌐"
                    type_label = "Web"
                else:
                    emoji = "📌"
                    type_label = "Source"

                # Trust level indicator
                trust_emoji = {"high": "✓", "medium": "○", "low": "△"}.get(
                    trust_level, "?"
                )

                content_parts.append(
                    f"- {emoji} **{type_label}** [{trust_emoji}]: {source_id}"
                )

            await cl.Message(
                content="\n".join(content_parts),
                author="System",
                parent_id=parent_id,
            ).send()

            cite_step.output = f"Displayed {len(citations)} citations"

    def _format_answer(self, response) -> str:
        """Format the answer with external source notice if needed."""
        answer = response.answer

        # Add external info notice if needed
        if response.external_info_used:
            if not answer.startswith("[Note:"):
                answer = (
                    "🌐 *This answer includes information from web search.*\n\n"
                    + answer
                )

        return answer

    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji for confidence level."""
        if confidence >= 0.8:
            return "🟢"
        elif confidence >= 0.6:
            return "🟡"
        elif confidence >= 0.4:
            return "🟠"
        else:
            return "🔴"

    async def close(self) -> None:
        """Clean up resources."""
        if self._agent:
            await self._agent.close()


class CitationDisplay:
    """Helper for displaying citations with formatting."""

    @staticmethod
    def format_citation(citation) -> str:
        """Format a single citation for display."""
        source_type = getattr(citation, 'source_type', 'unknown')
        source_id = getattr(citation, 'source_id', 'Unknown')
        excerpt = getattr(citation, 'excerpt', '')

        parts = [f"[{source_type.upper()}] {source_id}"]
        if excerpt:
            parts.append(f"  > {excerpt[:100]}...")

        return "\n".join(parts)

    @staticmethod
    def create_citation_elements(citations: List) -> List:
        """Create Chainlit elements for citations."""
        elements = []

        for i, cit in enumerate(citations):
            source_type = getattr(cit, 'source_type', 'unknown')
            source_id = getattr(cit, 'source_id', f'source_{i}')

            if source_type == "web_search":
                # For web sources, create a link if URL available
                if source_id.startswith("http"):
                    elements.append(
                        cl.Text(
                            name=f"source_{i}",
                            content=f"[{source_id}]({source_id})",
                            display="inline",
                        )
                    )

        return elements


class ReActStepVisualizer:
    """Helper for visualizing ReAct reasoning steps."""

    @staticmethod
    def format_thought(thought: str, max_length: int = 200) -> str:
        """Format a thought for display."""
        if len(thought) > max_length:
            return thought[:max_length] + "..."
        return thought

    @staticmethod
    def format_action(action) -> str:
        """Format an action for display."""
        if not action:
            return ""

        tool_name = getattr(action, 'tool_name', 'unknown')
        args = getattr(action, 'arguments', {})

        # Format arguments
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{tool_name}({arg_str})"

    @staticmethod
    def format_observation(observation: str, max_length: int = 150) -> str:
        """Format an observation for display."""
        if not observation:
            return ""
        if len(observation) > max_length:
            return observation[:max_length] + "..."
        return observation

    @staticmethod
    def create_step_markdown(steps: List) -> str:
        """Create markdown representation of steps."""
        parts = []

        for i, step in enumerate(steps, 1):
            step_parts = [f"### Step {i}"]

            thought = getattr(step, 'thought', '')
            if thought:
                step_parts.append(
                    f"**Thought:** {ReActStepVisualizer.format_thought(thought)}"
                )

            action = getattr(step, 'action', None)
            if action:
                step_parts.append(
                    f"**Action:** `{ReActStepVisualizer.format_action(action)}`"
                )

            observation = getattr(step, 'observation', '')
            if observation:
                step_parts.append(
                    f"**Observation:** {ReActStepVisualizer.format_observation(observation)}"
                )

            parts.append("\n".join(step_parts))

        return "\n\n---\n\n".join(parts)
