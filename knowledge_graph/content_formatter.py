"""
Content Formatter Module for Knowledge Graph Pipeline.

Provides three strategies for formatting raw webpage/article content:
1. SynthesizedHierarchical - Combine parent description + child subsections
2. AbstractOnly - Extract clean description per entity
3. LLMFormatted - Use LLM to create Wikipedia-style abstract

Usage:
    formatter = ContentFormatterFactory.create("synthesized", graph=graph, llm=llm)
    formatted_content = formatter.format(content, entity_context)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FormattingStrategy(str, Enum):
    """Available content formatting strategies."""
    RAW = "raw"                         # No formatting, use as-is
    SYNTHESIZED = "synthesized"         # Hierarchical doc with child subsections
    ABSTRACT_ONLY = "abstract"          # Extract clean abstract/description only
    LLM_FORMATTED = "llm"               # LLM-generated Wikipedia-style content


@dataclass
class EntityContext:
    """Context about an entity for content formatting."""
    qid: str                            # Wikidata QID (e.g., "Q123")
    name: str                           # Entity name
    description: Optional[str] = None   # Wikidata description
    wikipedia_url: Optional[str] = None

    # Relationships from the graph
    children: List[Dict[str, Any]] = field(default_factory=list)      # SUBCLASS_OF targets
    instances: List[Dict[str, Any]] = field(default_factory=list)     # INSTANCE_OF targets
    parts: List[Dict[str, Any]] = field(default_factory=list)         # PART_OF targets
    parent_of: List[Dict[str, Any]] = field(default_factory=list)     # Reverse relationships

    # Source content
    raw_content: Optional[str] = None   # Original scraped/fetched content


@dataclass
class FormattedContent:
    """Result of content formatting."""
    content: str                        # Formatted content
    abstract: Optional[str] = None      # Extracted abstract/summary
    sections: List[Dict[str, str]] = field(default_factory=list)  # {title, content} sections
    strategy_used: str = "raw"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentFormatter(ABC):
    """Abstract base class for content formatters."""

    def __init__(
        self,
        graph: Optional[Any] = None,
        llm: Optional[Any] = None,
        max_abstract_length: int = 500,
        max_section_length: int = 300,
    ):
        """
        Initialize the content formatter.

        Args:
            graph: FalkorDB graph connection for fetching relationships.
            llm: LLM instance for summarization (if needed).
            max_abstract_length: Maximum length for abstract/description.
            max_section_length: Maximum length for each section.
        """
        self.graph = graph
        self.llm = llm
        self.max_abstract_length = max_abstract_length
        self.max_section_length = max_section_length

    @abstractmethod
    def format(
        self,
        raw_content: str,
        context: EntityContext,
    ) -> FormattedContent:
        """
        Format raw content according to the strategy.

        Args:
            raw_content: Raw scraped/fetched content.
            context: Entity context with relationships.

        Returns:
            FormattedContent with processed content.
        """
        pass

    def _extract_first_paragraph(self, content: str, max_length: int = 500) -> str:
        """Extract the first meaningful paragraph from content."""
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content.strip())

        for para in paragraphs:
            para = para.strip()
            # Skip very short paragraphs (likely headers or noise)
            if len(para) < 50:
                continue
            # Skip paragraphs that look like navigation/headers
            if para.isupper() or para.startswith('#'):
                continue
            # Found a good paragraph
            if len(para) <= max_length:
                return para
            # Truncate at sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', para)
            result = ""
            for sent in sentences:
                if len(result) + len(sent) + 1 <= max_length:
                    result += sent + " "
                else:
                    break
            return result.strip() if result else para[:max_length] + "..."

        # Fallback: return truncated content
        return content[:max_length].strip() + "..." if len(content) > max_length else content.strip()

    def _clean_content(self, content: str) -> str:
        """Clean content by removing noise and normalizing whitespace."""
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # Remove common noise patterns
        noise_patterns = [
            r'Cookie\s*(Settings|Policy|Preferences).*?\n',
            r'Accept\s*(All)?\s*Cookies.*?\n',
            r'Skip to (main )?content.*?\n',
            r'Share this (page|article).*?\n',
            r'Print this page.*?\n',
            r'Follow us on.*?\n',
            r'Subscribe to our newsletter.*?\n',
            r'©\s*\d{4}.*?\n',
            r'All rights reserved.*?\n',
        ]

        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        return content.strip()

    def fetch_entity_context(self, qid: str) -> EntityContext:
        """
        Fetch entity context including relationships from the graph.

        Args:
            qid: Wikidata QID to fetch context for.

        Returns:
            EntityContext with entity info and relationships.
        """
        if not self.graph:
            return EntityContext(qid=qid, name=qid)

        try:
            # Fetch entity info
            entity_query = f"""
                MATCH (w:WikiPage {{wikidata_id: '{qid}'}})
                RETURN w.name as name, w.description as description,
                       w.wikipedia_url as wikipedia_url
            """
            result = self.graph.query(entity_query)

            if not result.result_set:
                return EntityContext(qid=qid, name=qid)

            row = result.result_set[0]
            context = EntityContext(
                qid=qid,
                name=row[0] or qid,
                description=row[1],
                wikipedia_url=row[2],
            )

            # Fetch children (things that are SUBCLASS_OF this entity)
            children_query = f"""
                MATCH (child:WikiPage)-[:SUBCLASS_OF]->(w:WikiPage {{wikidata_id: '{qid}'}})
                RETURN child.wikidata_id as qid, child.name as name,
                       child.description as description
                LIMIT 20
            """
            children_result = self.graph.query(children_query)
            context.children = [
                {"qid": r[0], "name": r[1], "description": r[2]}
                for r in children_result.result_set
            ]

            # Fetch instances (things that are INSTANCE_OF this entity)
            instances_query = f"""
                MATCH (inst:WikiPage)-[:INSTANCE_OF]->(w:WikiPage {{wikidata_id: '{qid}'}})
                RETURN inst.wikidata_id as qid, inst.name as name,
                       inst.description as description
                LIMIT 20
            """
            instances_result = self.graph.query(instances_query)
            context.instances = [
                {"qid": r[0], "name": r[1], "description": r[2]}
                for r in instances_result.result_set
            ]

            # Fetch parts (things that are PART_OF this entity)
            parts_query = f"""
                MATCH (part:WikiPage)-[:PART_OF]->(w:WikiPage {{wikidata_id: '{qid}'}})
                RETURN part.wikidata_id as qid, part.name as name,
                       part.description as description
                LIMIT 20
            """
            parts_result = self.graph.query(parts_query)
            context.parts = [
                {"qid": r[0], "name": r[1], "description": r[2]}
                for r in parts_result.result_set
            ]

            return context

        except Exception as e:
            logger.warning(f"Failed to fetch entity context for {qid}: {e}")
            return EntityContext(qid=qid, name=qid)


class RawFormatter(ContentFormatter):
    """No-op formatter that returns content as-is (with basic cleaning)."""

    def format(self, raw_content: str, context: EntityContext) -> FormattedContent:
        """Return cleaned but unformatted content."""
        cleaned = self._clean_content(raw_content)
        return FormattedContent(
            content=cleaned,
            abstract=self._extract_first_paragraph(cleaned, self.max_abstract_length),
            strategy_used="raw",
        )


class SynthesizedHierarchicalFormatter(ContentFormatter):
    """
    Creates a structured document combining entity description with child subsections.

    Output format:
    ```
    # Entity Name

    [Abstract/description paragraph]

    ## Types (Subclasses)
    - **Child Name**: Child description...

    ## Instances
    - **Instance Name**: Instance description...

    ## Components (Parts)
    - **Part Name**: Part description...
    ```
    """

    def format(self, raw_content: str, context: EntityContext) -> FormattedContent:
        """Create hierarchical document with subsections."""
        cleaned_content = self._clean_content(raw_content)

        # Extract or use provided abstract
        if context.description:
            abstract = context.description
        else:
            abstract = self._extract_first_paragraph(cleaned_content, self.max_abstract_length)

        sections = []
        content_parts = []

        # Header
        content_parts.append(f"# {context.name}\n")
        content_parts.append(f"{abstract}\n")

        # Add subclasses section if any
        if context.children:
            section_content = self._format_entity_list(context.children, "Types")
            content_parts.append(section_content)
            sections.append({"title": "Types (Subclasses)", "content": section_content})

        # Add instances section if any
        if context.instances:
            section_content = self._format_entity_list(context.instances, "Instances")
            content_parts.append(section_content)
            sections.append({"title": "Instances", "content": section_content})

        # Add parts section if any
        if context.parts:
            section_content = self._format_entity_list(context.parts, "Components")
            content_parts.append(section_content)
            sections.append({"title": "Components (Parts)", "content": section_content})

        # Combine all parts
        formatted = "\n".join(content_parts)

        return FormattedContent(
            content=formatted,
            abstract=abstract,
            sections=sections,
            strategy_used="synthesized",
            metadata={
                "num_children": len(context.children),
                "num_instances": len(context.instances),
                "num_parts": len(context.parts),
            },
        )

    def _format_entity_list(self, entities: List[Dict], section_title: str) -> str:
        """Format a list of entities as a markdown section."""
        lines = [f"\n## {section_title}\n"]

        for entity in entities:
            name = entity.get("name", entity.get("qid", "Unknown"))
            description = entity.get("description", "")

            if description:
                # Truncate long descriptions
                if len(description) > self.max_section_length:
                    description = description[:self.max_section_length].rsplit(' ', 1)[0] + "..."
                lines.append(f"- **{name}**: {description}")
            else:
                lines.append(f"- **{name}**")

        return "\n".join(lines)


class AbstractOnlyFormatter(ContentFormatter):
    """
    Extracts only the abstract/description for each entity.

    Structure comes from graph relationships, not the document.
    Each entity gets a clean, concise description.
    """

    def __init__(
        self,
        graph: Optional[Any] = None,
        llm: Optional[Any] = None,
        max_abstract_length: int = 500,
        max_section_length: int = 300,
        extract_from_content: bool = True,
    ):
        """
        Initialize AbstractOnlyFormatter.

        Args:
            extract_from_content: If True, extract abstract from content if
                                  no description is available from Wikidata.
        """
        super().__init__(graph, llm, max_abstract_length, max_section_length)
        self.extract_from_content = extract_from_content

    def format(self, raw_content: str, context: EntityContext) -> FormattedContent:
        """Extract clean abstract/description only."""
        cleaned_content = self._clean_content(raw_content)

        # Priority: 1) Wikidata description, 2) Extracted from content
        if context.description and len(context.description) >= 50:
            abstract = context.description
        elif self.extract_from_content:
            abstract = self._extract_abstract(cleaned_content, context.name)
        else:
            abstract = context.description or ""

        # Format as simple titled content
        formatted = f"# {context.name}\n\n{abstract}"

        return FormattedContent(
            content=formatted,
            abstract=abstract,
            strategy_used="abstract",
            metadata={
                "source": "wikidata" if context.description else "extracted",
                "original_length": len(raw_content),
                "abstract_length": len(abstract),
            },
        )

    def _extract_abstract(self, content: str, entity_name: str) -> str:
        """
        Extract the most relevant abstract paragraph from content.

        Looks for paragraphs that mention the entity name or seem definitional.
        """
        paragraphs = re.split(r'\n\s*\n', content.strip())

        # Score paragraphs by relevance
        scored_paragraphs: List[Tuple[float, str]] = []

        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            if para.isupper():
                continue

            score = 0.0
            para_lower = para.lower()
            name_lower = entity_name.lower()

            # Higher score if entity name appears
            if name_lower in para_lower:
                score += 3.0
                # Even higher if it's at the beginning (definitional)
                if para_lower.startswith(name_lower) or para_lower.startswith(f"a {name_lower}") or \
                   para_lower.startswith(f"an {name_lower}") or para_lower.startswith(f"the {name_lower}"):
                    score += 2.0

            # Definitional patterns
            definitional_patterns = [
                r'\bis\s+a\b', r'\bare\s+a\b', r'\brefers\s+to\b',
                r'\bis\s+defined\b', r'\bis\s+the\b', r'\bare\s+the\b',
            ]
            for pattern in definitional_patterns:
                if re.search(pattern, para_lower):
                    score += 1.0

            # Penalize if it looks like navigation or boilerplate
            if any(x in para_lower for x in ['click here', 'learn more', 'contact us', 'subscribe']):
                score -= 2.0

            # Prefer paragraphs that are good length (100-400 chars)
            if 100 <= len(para) <= 400:
                score += 0.5

            scored_paragraphs.append((score, para))

        # Sort by score (descending) and take best
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])

        if scored_paragraphs:
            best_para = scored_paragraphs[0][1]
            # Truncate if needed
            if len(best_para) > self.max_abstract_length:
                # Try to cut at sentence boundary
                sentences = re.split(r'(?<=[.!?])\s+', best_para)
                result = ""
                for sent in sentences:
                    if len(result) + len(sent) + 1 <= self.max_abstract_length:
                        result += sent + " "
                    else:
                        break
                return result.strip() if result else best_para[:self.max_abstract_length] + "..."
            return best_para

        # Fallback
        return self._extract_first_paragraph(content, self.max_abstract_length)


class LLMFormattedFormatter(ContentFormatter):
    """
    Uses an LLM to create a clean, Wikipedia-style abstract from raw content.

    The LLM is prompted to:
    1. Extract key information about the entity
    2. Write a concise, encyclopedic summary
    3. Maintain factual accuracy without adding information
    """

    SYSTEM_PROMPT = """You are a technical writer creating encyclopedic content.
Your task is to rewrite the provided content into a clean, Wikipedia-style article.

Guidelines:
- Write in a neutral, encyclopedic tone
- Start with a clear definition of what the subject is
- Include key facts and technical details
- Do not add information not present in the source
- Do not include marketing language or promotional content
- Keep the response concise and focused
- Use proper technical terminology"""

    USER_PROMPT_TEMPLATE = """Please create a clean, Wikipedia-style summary for "{entity_name}" based on the following content.
Focus on the most important and factual information.

Source content:
{content}

Write a concise encyclopedic summary (2-4 paragraphs, max {max_length} characters):"""

    def format(self, raw_content: str, context: EntityContext) -> FormattedContent:
        """Use LLM to create Wikipedia-style content."""
        cleaned_content = self._clean_content(raw_content)

        # If no LLM available, fall back to abstract extraction
        if not self.llm:
            logger.warning("No LLM provided for LLMFormattedFormatter, falling back to abstract extraction")
            fallback = AbstractOnlyFormatter(
                graph=self.graph,
                max_abstract_length=self.max_abstract_length,
            )
            result = fallback.format(raw_content, context)
            result.strategy_used = "llm_fallback"
            return result

        try:
            # Truncate content if too long for LLM context
            max_input_length = 4000  # Leave room for prompt and response
            if len(cleaned_content) > max_input_length:
                cleaned_content = cleaned_content[:max_input_length] + "..."

            # Build prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                entity_name=context.name,
                content=cleaned_content,
                max_length=self.max_abstract_length,
            )

            # Call LLM
            response = self._call_llm(user_prompt)

            if response:
                # Clean up LLM response
                formatted_content = self._clean_llm_response(response)

                # Add title
                full_content = f"# {context.name}\n\n{formatted_content}"

                return FormattedContent(
                    content=full_content,
                    abstract=self._extract_first_paragraph(formatted_content, self.max_abstract_length),
                    strategy_used="llm",
                    metadata={
                        "llm_response_length": len(response),
                        "input_length": len(cleaned_content),
                    },
                )
            else:
                raise ValueError("Empty LLM response")

        except Exception as e:
            logger.warning(f"LLM formatting failed for {context.name}: {e}")
            # Fall back to abstract extraction
            fallback = AbstractOnlyFormatter(
                graph=self.graph,
                max_abstract_length=self.max_abstract_length,
            )
            result = fallback.format(raw_content, context)
            result.strategy_used = "llm_fallback"
            result.metadata["llm_error"] = str(e)
            return result

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # Support different LLM interfaces
        if hasattr(self.llm, 'invoke'):
            # LangChain-style LLM
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        elif hasattr(self.llm, 'generate'):
            # Ollama or similar
            response = self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
            return response
        elif callable(self.llm):
            # Simple callable
            return self.llm(prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {type(self.llm)}")

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response."""
        # Remove common prefixes
        prefixes_to_remove = [
            "Here is", "Here's", "Based on the provided content",
            "According to the source", "The following is",
        ]

        response = response.strip()
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                # Find the end of the sentence and remove
                match = re.match(rf'{re.escape(prefix)}[^.]*[.:]?\s*', response, re.IGNORECASE)
                if match:
                    response = response[match.end():]

        # Remove trailing notes
        trailing_patterns = [
            r'\n\nNote:.*$',
            r'\n\nPlease note.*$',
            r'\n\nThis summary.*$',
        ]
        for pattern in trailing_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)

        return response.strip()


class ContentFormatterFactory:
    """Factory for creating content formatters."""

    _formatters = {
        FormattingStrategy.RAW: RawFormatter,
        FormattingStrategy.SYNTHESIZED: SynthesizedHierarchicalFormatter,
        FormattingStrategy.ABSTRACT_ONLY: AbstractOnlyFormatter,
        FormattingStrategy.LLM_FORMATTED: LLMFormattedFormatter,
    }

    @classmethod
    def create(
        cls,
        strategy: str | FormattingStrategy,
        graph: Optional[Any] = None,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> ContentFormatter:
        """
        Create a content formatter for the given strategy.

        Args:
            strategy: Formatting strategy name or enum.
            graph: FalkorDB graph connection.
            llm: LLM instance for summarization.
            **kwargs: Additional arguments passed to formatter.

        Returns:
            ContentFormatter instance.
        """
        if isinstance(strategy, str):
            try:
                strategy = FormattingStrategy(strategy.lower())
            except ValueError:
                logger.warning(f"Unknown formatting strategy '{strategy}', using 'raw'")
                strategy = FormattingStrategy.RAW

        formatter_class = cls._formatters.get(strategy, RawFormatter)
        return formatter_class(graph=graph, llm=llm, **kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        """Return list of available strategy names."""
        return [s.value for s in FormattingStrategy]
