"""
Chainlit frontend application for GraphRAG knowledge base.

Features:
- Interactive chat with knowledge graph
- Chain-of-thought step visualization
- PyVis graph rendering
- Human-in-the-loop entity disambiguation
- Tool use display
- Q&A Mode: ReAct agent with hybrid GraphRAG + web search
- Research Mode: Autonomous gap-filling with approval workflow

Run with: chainlit run frontend/app.py --port 8000
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Literal

# Add parent directory to path for agent imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl

from integrations.graphrag_integration import ChainlitGraphRAG
from components.graph_visualizer import GraphVisualizer, create_graph_legend
from components.disambiguation_ui import DisambiguationHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mode type
ModeType = Literal["classic", "qa", "research"]

# Global instances
rag: Optional[ChainlitGraphRAG] = None
visualizer: Optional[GraphVisualizer] = None
disambiguator: Optional[DisambiguationHandler] = None


# Mode settings for Chainlit
@cl.set_chat_profiles
async def chat_profiles():
    """Define available chat profiles (modes)."""
    return [
        cl.ChatProfile(
            name="Q&A Agent",
            markdown_description="ReAct agent with hybrid GraphRAG + web search. Best for complex questions.",
            icon="https://cdn-icons-png.flaticon.com/512/1998/1998614.png",
            default=True,
        ),
        cl.ChatProfile(
            name="Classic",
            markdown_description="Classic GraphRAG mode with entity extraction and visualization.",
            icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        ),
        cl.ChatProfile(
            name="Research",
            markdown_description="Autonomous research mode to fill knowledge gaps. Use 'start research' to begin.",
            icon="https://cdn-icons-png.flaticon.com/512/2920/2920277.png",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize session resources based on selected mode."""
    global rag, visualizer, disambiguator

    # Get selected profile
    chat_profile = cl.user_session.get("chat_profile")
    mode = _profile_to_mode(chat_profile)
    cl.user_session.set("mode", mode)

    # Initialize GraphRAG (needed for all modes)
    try:
        rag = ChainlitGraphRAG()
        logger.info("GraphRAG initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
        await cl.Message(
            content=f"Failed to connect to knowledge graph: {e}",
            author="System",
        ).send()
        return

    # Initialize visualizer
    visualizer = GraphVisualizer(height="350px")

    # Initialize disambiguator
    disambiguator = DisambiguationHandler()

    # Store in session
    cl.user_session.set("rag", rag)
    cl.user_session.set("visualizer", visualizer)
    cl.user_session.set("disambiguator", disambiguator)

    # Initialize mode-specific handlers
    if mode == "qa":
        await _init_qa_mode()
    elif mode == "research":
        await _init_research_mode()
    else:
        await _init_classic_mode()


def _profile_to_mode(profile: Optional[str]) -> ModeType:
    """Convert chat profile name to mode type."""
    if profile == "Q&A Agent":
        return "qa"
    elif profile == "Research":
        return "research"
    return "classic"


async def _init_classic_mode():
    """Initialize classic GraphRAG mode."""
    await cl.Message(
        content=(
            "Welcome to the **Knowledge Graph Assistant**!\n\n"
            "I can help you explore and query the knowledge graph. Ask me questions about:\n"
            "- Entities and their relationships\n"
            "- Connections between concepts\n"
            "- Historical or factual information in the graph\n\n"
            "I'll show you the relevant graph data and my reasoning process."
        ),
        author="Assistant",
    ).send()


async def _init_qa_mode():
    """Initialize Q&A agent mode."""
    try:
        from modes.qa_mode import QAModeHandler
        handler = QAModeHandler()
        success = await handler.initialize()
        if success:
            cl.user_session.set("qa_handler", handler)
            await cl.Message(
                content=(
                    "Welcome to **Q&A Agent Mode**!\n\n"
                    "I use a ReAct reasoning loop to answer your questions:\n"
                    "1. First I search the knowledge graph\n"
                    "2. If needed, I search the web for additional information\n"
                    "3. I combine sources and provide cited answers\n\n"
                    "Ask me anything! I'll show my reasoning process and cite my sources."
                ),
                author="Q&A Agent",
            ).send()
        else:
            await cl.Message(
                content="Q&A Agent initialization failed. Falling back to classic mode.",
                author="System",
            ).send()
            cl.user_session.set("mode", "classic")
    except ImportError as e:
        logger.error(f"Q&A mode import failed: {e}", exc_info=True)
        await cl.Message(
            content=f"Q&A Agent mode not available: {e}. Using classic mode.",
            author="System",
        ).send()
        cl.user_session.set("mode", "classic")
    except Exception as e:
        logger.error(f"Q&A mode initialization error: {e}", exc_info=True)
        await cl.Message(
            content=f"Q&A Agent error: {e}. Using classic mode.",
            author="System",
        ).send()
        cl.user_session.set("mode", "classic")


async def _init_research_mode():
    """Initialize research mode."""
    try:
        from modes.research_mode import ResearchModeHandler
        handler = ResearchModeHandler()
        success = await handler.initialize()
        if success:
            cl.user_session.set("research_handler", handler)
            await cl.Message(
                content=(
                    "Welcome to **Research Mode**!\n\n"
                    "I can help identify gaps in the knowledge graph and research new topics.\n\n"
                    "**Commands:**\n"
                    "- `start research` - Detect gaps and generate research topics\n"
                    "- `approve 1,2,4,7` - Approve specific topics (numbered 1-10)\n"
                    "- `approve all` - Approve all topics\n"
                    "- `stop` - Terminate research gracefully\n"
                    "- `status` - Show current status\n"
                    "- `set time 30` - Set time limit in minutes\n\n"
                    "Type `start research` to begin!"
                ),
                author="Research Assistant",
            ).send()
        else:
            await cl.Message(
                content="Research mode initialization failed. Falling back to classic mode.",
                author="System",
            ).send()
            cl.user_session.set("mode", "classic")
    except ImportError as e:
        logger.warning(f"Research mode not available: {e}")
        await cl.Message(
            content="Research mode not available. Using classic mode.",
            author="System",
        ).send()
        cl.user_session.set("mode", "classic")


@cl.on_chat_end
async def on_chat_end():
    """Clean up session resources."""
    # Clean up mode handlers
    qa_handler = cl.user_session.get("qa_handler")
    if qa_handler:
        await qa_handler.close()

    research_handler = cl.user_session.get("research_handler")
    if research_handler:
        await research_handler.close()

    rag = cl.user_session.get("rag")
    if rag:
        rag.close()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages based on mode."""
    mode = cl.user_session.get("mode", "classic")

    if mode == "qa":
        await _handle_qa_message(message)
    elif mode == "research":
        await _handle_research_message(message)
    else:
        await _handle_classic_message(message)


async def _handle_qa_message(message: cl.Message):
    """Handle message in Q&A mode."""
    handler = cl.user_session.get("qa_handler")

    if handler:
        await handler.handle_message(message)
    else:
        # Fall back to classic mode
        await _handle_classic_message(message)


async def _handle_research_message(message: cl.Message):
    """Handle message in Research mode."""
    handler = cl.user_session.get("research_handler")

    if handler:
        await handler.handle_message(message)
    else:
        # Fall back to classic mode
        await _handle_classic_message(message)


async def _handle_classic_message(message: cl.Message):
    """Handle message in classic GraphRAG mode."""
    query = message.content

    # Get session resources
    rag = cl.user_session.get("rag")
    visualizer = cl.user_session.get("visualizer")
    disambiguator = cl.user_session.get("disambiguator")

    if not rag:
        await cl.Message(
            content="Knowledge graph connection not available. Please refresh the page.",
            author="System",
        ).send()
        return

    # Create parent step for the full process
    async with cl.Step(name="Processing Query", type="run") as parent_step:
        parent_step.input = query

        # Step 1: Entity Extraction
        async with cl.Step(name="Entity Extraction", type="tool") as extract_step:
            extract_step.input = query
            entities = await rag.extract_entities(query)
            extract_step.output = f"Found {len(entities)} entities: {[e.get('name') for e in entities]}"

            if entities:
                entity_list = ", ".join([f"**{e.get('name')}** ({e.get('type', 'Entity')})" for e in entities])
                await cl.Message(
                    content=f"Identified entities: {entity_list}",
                    author="System",
                    parent_id=parent_step.id,
                ).send()

        # Step 2: Entity Resolution (with potential disambiguation)
        resolved_entities = []
        if entities and disambiguator:
            async with cl.Step(name="Entity Resolution", type="tool") as resolve_step:
                resolve_step.input = f"Resolving {len(entities)} entities"

                for entity in entities:
                    name = entity.get("name", "")
                    if not name:
                        continue

                    candidates = await rag.resolve_entity(name)

                    if not candidates:
                        resolve_step.output = f"No matches found for '{name}'"
                    elif len(candidates) == 1:
                        resolved_entities.append(candidates[0])
                    elif disambiguator.needs_disambiguation(candidates):
                        # Show disambiguation UI
                        selected = await disambiguator.ask_disambiguation(name, candidates)
                        if selected:
                            resolved_entities.append(selected)

                resolve_step.output = f"Resolved {len(resolved_entities)} entities"

        # Step 3: Graph Retrieval
        async with cl.Step(name="Graph Retrieval", type="retrieval") as retrieval_step:
            retrieval_step.input = query

            context_data = await rag.retrieve_graph_context(query)
            retrieval_step.output = f"Retrieved {len(context_data)} relevant entities from graph"

            if context_data:
                # Show retrieved entities summary
                entity_summary = []
                for entry in context_data[:5]:
                    name = entry.get("entity", "Unknown")
                    etype = entry.get("type", "")
                    score = entry.get("score", 0)
                    entity_summary.append(f"- **{name}** ({etype}) - {score:.0%} relevance")

                await cl.Message(
                    content="**Retrieved from Knowledge Graph:**\n" + "\n".join(entity_summary),
                    author="System",
                    parent_id=parent_step.id,
                ).send()

        # Step 4: Graph Visualization
        if context_data and visualizer:
            async with cl.Step(name="Graph Visualization", type="tool") as viz_step:
                viz_step.input = f"Visualizing {len(context_data)} entities"

                try:
                    # Create graph HTML
                    graph_html = visualizer.create_graph(context_data)

                    if graph_html:
                        # Save to temp file for Chainlit
                        import tempfile
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".html", delete=False, dir="/tmp"
                        ) as f:
                            f.write(graph_html)
                            temp_path = f.name

                        # Send as file element
                        elements = [
                            cl.File(
                                name="knowledge_graph.html",
                                path=temp_path,
                                display="inline",
                            )
                        ]

                        await cl.Message(
                            content="**Knowledge Graph Visualization:**\n" + create_graph_legend(),
                            elements=elements,
                            author="System",
                            parent_id=parent_step.id,
                        ).send()

                        viz_step.output = "Graph visualization created"

                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                    else:
                        viz_step.output = "Visualization unavailable"

                except Exception as e:
                    logger.error(f"Visualization failed: {e}")
                    viz_step.output = f"Visualization error: {e}"

        # Step 5: Answer Generation
        async with cl.Step(name="Answer Generation", type="llm") as gen_step:
            gen_step.input = f"Query: {query}\nContext: {len(context_data)} entities"

            # Show thinking indicator
            msg = cl.Message(content="", author="Assistant")
            await msg.send()

            # Generate answer
            answer = await rag.generate_answer(query, context_data)

            # Stream the answer (simulated since we already have it)
            await msg.stream_token(answer)
            await msg.update()

            gen_step.output = f"Generated {len(answer)} character response"

        # Set parent step output
        parent_step.output = "Query processed successfully"


@cl.action_callback("select_entity_*")
async def handle_entity_selection(action: cl.Action):
    """Handle entity selection from disambiguation UI."""
    selected_name = action.value
    await cl.Message(
        content=f"Selected: **{selected_name}**",
        author="System",
    ).send()
    return selected_name


@cl.action_callback("cancel_disambiguation")
async def handle_cancel(action: cl.Action):
    """Handle disambiguation cancellation."""
    await cl.Message(
        content="Skipped entity disambiguation.",
        author="System",
    ).send()
    return None


# Entry point for direct execution
if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
