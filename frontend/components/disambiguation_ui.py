"""
Human-in-the-loop disambiguation UI for Chainlit.

Provides interactive entity disambiguation when multiple
candidates are found in the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional

import chainlit as cl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisambiguationHandler:
    """
    Handle entity disambiguation with user interaction.

    When an entity name is ambiguous (multiple matches in the graph),
    presents the user with candidate options to select from.
    """

    def __init__(self, min_candidates_for_disambiguation: int = 2):
        """
        Args:
            min_candidates_for_disambiguation: Minimum candidates to trigger disambiguation
        """
        self.min_candidates = min_candidates_for_disambiguation

    def needs_disambiguation(
        self,
        candidates: List[Dict[str, Any]],
    ) -> bool:
        """Check if disambiguation is needed."""
        return len(candidates) >= self.min_candidates

    async def create_disambiguation_actions(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
    ) -> List[cl.Action]:
        """
        Create Chainlit actions for disambiguation choices.

        Args:
            entity_name: Original ambiguous name
            candidates: List of candidate entities

        Returns:
            List of Chainlit Action objects
        """
        actions = []

        for i, candidate in enumerate(candidates[:5]):  # Limit to 5 options
            name = candidate.get("name", f"Option {i+1}")
            entity_type = candidate.get("type", "Entity")
            description = candidate.get("description", "")
            score = candidate.get("score", 0)

            # Create button label
            label = f"{name} ({entity_type})"
            if score:
                label += f" - {score:.0%} match"

            # Create action
            action = cl.Action(
                name=f"select_entity_{i}",
                label=label,
                value=name,
                description=description[:100] if description else "No description",
            )
            actions.append(action)

        return actions

    async def ask_disambiguation(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Ask user to disambiguate between candidates.

        Args:
            entity_name: Original ambiguous name
            candidates: List of candidate entities

        Returns:
            Selected entity dict, or None if cancelled
        """
        if not self.needs_disambiguation(candidates):
            # Return best match if only one
            return candidates[0] if candidates else None

        # Create actions for each candidate
        actions = await self.create_disambiguation_actions(entity_name, candidates)

        # Add cancel option
        actions.append(
            cl.Action(
                name="cancel_disambiguation",
                label="Skip this entity",
                value="__cancel__",
            )
        )

        # Create message with description of candidates
        candidate_descriptions = []
        for i, c in enumerate(candidates[:5]):
            desc = c.get("description", "No description available")
            if len(desc) > 100:
                desc = desc[:100] + "..."
            candidate_descriptions.append(
                f"**{i+1}. {c.get('name', 'Unknown')}** ({c.get('type', 'Entity')})\n"
                f"   {desc}"
            )

        message_content = (
            f"The name **\"{entity_name}\"** could refer to multiple entities:\n\n"
            + "\n\n".join(candidate_descriptions)
            + "\n\nPlease select the correct entity:"
        )

        # Ask user with actions
        response = await cl.AskActionMessage(
            content=message_content,
            actions=actions,
        ).send()

        if response is None:
            return None

        selected_value = response.get("value", "")

        if selected_value == "__cancel__":
            return None

        # Find selected candidate
        for candidate in candidates:
            if candidate.get("name") == selected_value:
                return candidate

        return None

    async def resolve_ambiguous_entities(
        self,
        entities: List[Dict[str, Any]],
        resolve_func,
    ) -> List[Dict[str, Any]]:
        """
        Resolve a list of potentially ambiguous entities.

        Args:
            entities: List of extracted entity mentions
            resolve_func: Async function to resolve entity name to candidates

        Returns:
            List of resolved entity dicts
        """
        resolved = []

        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue

            # Get candidates from graph
            candidates = await resolve_func(name)

            if not candidates:
                # No matches found
                await cl.Message(
                    content=f"Could not find **\"{name}\"** in the knowledge graph.",
                    author="System",
                ).send()
                continue

            if len(candidates) == 1:
                # Exact match
                resolved.append(candidates[0])
            else:
                # Disambiguation needed
                selected = await self.ask_disambiguation(name, candidates)
                if selected:
                    resolved.append(selected)

        return resolved


async def create_entity_chip(entity: Dict[str, Any]) -> str:
    """
    Create an HTML chip for displaying an entity.

    Args:
        entity: Entity dict with name, type, description

    Returns:
        HTML string for the chip
    """
    name = entity.get("name", "Unknown")
    entity_type = entity.get("type", "Entity")

    # Color based on type
    colors = {
        "Person": "#ff6b6b",
        "Organization": "#4ecdc4",
        "Concept": "#45b7d1",
        "Event": "#96ceb4",
        "Location": "#ffeaa7",
    }
    color = colors.get(entity_type, "#74b9ff")

    return (
        f'<span style="display: inline-block; padding: 4px 8px; '
        f'background-color: {color}20; border: 1px solid {color}; '
        f'border-radius: 16px; margin: 2px; font-size: 12px;">'
        f'{name} <span style="color: #666;">({entity_type})</span></span>'
    )
