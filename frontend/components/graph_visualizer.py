"""
PyVis graph visualization component for Chainlit.

Creates interactive network visualizations from knowledge graph context.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("PyVis not installed. Install with: pip install pyvis")


# Node color scheme by entity type
NODE_COLORS = {
    "Person": "#ff6b6b",      # Red
    "Organization": "#4ecdc4", # Teal
    "Concept": "#45b7d1",     # Blue
    "Event": "#96ceb4",       # Green
    "Location": "#ffeaa7",    # Yellow
    "Theory": "#dfe6e9",      # Light gray
    "Law": "#a29bfe",         # Purple
    "Document": "#fd79a8",    # Pink
    "default": "#74b9ff",     # Light blue (fallback)
}


def get_node_color(entity_type: str) -> str:
    """Get color for entity type."""
    return NODE_COLORS.get(entity_type, NODE_COLORS["default"])


class GraphVisualizer:
    """
    Create PyVis visualizations from graph context data.

    Features:
    - Color-coded nodes by entity type
    - Edge labels with relationship types
    - Interactive physics-based layout
    - Hover tooltips with descriptions
    """

    def __init__(
        self,
        height: str = "400px",
        width: str = "100%",
        bgcolor: str = "#ffffff",
        font_color: str = "#333333",
    ):
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color

    def create_graph(
        self,
        context_data: List[Dict[str, Any]],
        notebook: bool = False,
    ) -> Optional[str]:
        """
        Create PyVis network visualization from context data.

        Args:
            context_data: List of context entries from GraphRAG
            notebook: Whether to render for Jupyter notebook

        Returns:
            HTML string of the visualization, or None if PyVis unavailable
        """
        if not PYVIS_AVAILABLE:
            logger.warning("PyVis not available, cannot create graph")
            return None

        if not context_data:
            return None

        # Create network
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            notebook=notebook,
            directed=True,
        )

        # Physics settings for better layout
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            },
            "nodes": {
                "font": {"size": 14},
                "borderWidth": 2,
                "shadow": true
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "font": {"size": 10, "align": "middle"},
                "smooth": {"type": "curvedCW", "roundness": 0.2}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200
            }
        }
        """)

        # Track added nodes to avoid duplicates
        added_nodes = set()

        # Add nodes and edges
        for entry in context_data:
            entity_name = entry.get("entity", "")
            entity_type = entry.get("type", "Entity")
            description = entry.get("description", "")
            score = entry.get("score", 0)

            if not entity_name:
                continue

            # Add main entity node
            if entity_name not in added_nodes:
                net.add_node(
                    entity_name,
                    label=entity_name,
                    title=f"{entity_type}\n{description[:200]}..." if len(description) > 200 else f"{entity_type}\n{description}",
                    color=get_node_color(entity_type),
                    size=20 + (score * 10),  # Size based on relevance
                    shape="dot",
                )
                added_nodes.add(entity_name)

            # Add relationships
            relationships = entry.get("relationships", [])
            for rel in relationships:
                if not rel.get("relation"):
                    continue

                target = rel.get("target", "")
                target_type = rel.get("target_type", "Entity")
                rel_type = rel.get("relation", "RELATED_TO")
                rel_description = rel.get("description", "")

                if not target:
                    continue

                # Add target node
                if target not in added_nodes:
                    net.add_node(
                        target,
                        label=target,
                        title=f"{target_type}",
                        color=get_node_color(target_type),
                        size=15,
                        shape="dot",
                    )
                    added_nodes.add(target)

                # Add edge
                edge_title = f"{rel_type}"
                if rel_description:
                    edge_title += f"\n{rel_description[:100]}..."

                net.add_edge(
                    entity_name,
                    target,
                    title=edge_title,
                    label=rel_type.replace("_", " ").lower(),
                    color="#888888",
                )

        # Generate HTML
        try:
            # Create temporary file for HTML
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                net.save_graph(f.name)
                with open(f.name, "r") as html_file:
                    html_content = html_file.read()
                os.unlink(f.name)
                return html_content
        except Exception as e:
            logger.error(f"Failed to generate graph HTML: {e}")
            return None

    def create_embedded_html(
        self,
        context_data: List[Dict[str, Any]],
        title: str = "Knowledge Graph",
    ) -> str:
        """
        Create HTML that can be embedded in Chainlit messages.

        Args:
            context_data: Graph context data
            title: Title for the visualization

        Returns:
            HTML string suitable for embedding
        """
        graph_html = self.create_graph(context_data)

        if not graph_html:
            return f"<p><em>Graph visualization unavailable</em></p>"

        # Extract just the body content for embedding
        # PyVis generates a full HTML document, we need to extract the relevant parts
        try:
            # Find the script and div content
            import re

            # Extract the network container div
            div_match = re.search(r'<div id="mynetwork".*?</div>', graph_html, re.DOTALL)
            div_content = div_match.group(0) if div_match else ""

            # Extract the vis.js script initialization
            script_match = re.search(r'<script[^>]*>.*?new vis\.Network.*?</script>', graph_html, re.DOTALL)
            script_content = script_match.group(0) if script_match else ""

            # Create embedded HTML
            embedded = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0;">{title}</h4>
                {div_content}
                {script_content}
            </div>
            """

            return embedded

        except Exception as e:
            logger.error(f"Failed to create embedded HTML: {e}")
            return f"<p><em>Graph visualization error: {e}</em></p>"


def create_pyvis_graph(
    context_data: List[Dict[str, Any]],
    height: str = "400px",
) -> str:
    """
    Convenience function to create a PyVis graph.

    Args:
        context_data: Graph context from GraphRAG
        height: Height of the visualization

    Returns:
        HTML string of the visualization
    """
    visualizer = GraphVisualizer(height=height)
    return visualizer.create_graph(context_data) or ""


def create_graph_legend() -> str:
    """Create HTML legend for graph node colors."""
    legend_items = []
    for entity_type, color in NODE_COLORS.items():
        if entity_type != "default":
            legend_items.append(
                f'<span style="display: inline-block; margin-right: 15px;">'
                f'<span style="display: inline-block; width: 12px; height: 12px; '
                f'background-color: {color}; border-radius: 50%; margin-right: 5px;"></span>'
                f'{entity_type}</span>'
            )

    return (
        '<div style="font-size: 12px; color: #666; margin-top: 5px;">'
        f'{"".join(legend_items)}'
        '</div>'
    )
