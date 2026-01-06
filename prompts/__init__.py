"""
System prompts for the LLM2KG agents.

This module contains all system prompts used in the knowledge graph extraction
and processing pipeline.
"""

from .agent_skb_prompts import (
    KG_EXTRACTION_PROMPT,
    KG_EXTRACTION_PROMPT_MINIMAL,
    KG_EXTRACTION_PROMPT_COMPACT,
    DEFAULT_EXTRACTION_PROMPT,
    get_extraction_prompt,
    ONTOLOGY_MAPPING_PROMPT_TEMPLATE,
    FROZEN_ONTOLOGY_MAPPING_PROMPT_TEMPLATE,
    LABEL_GENERALIZATION_PROMPT_TEMPLATE,
    ENTITY_VERIFICATION_PROMPT,
    LABEL_VERIFICATION_PROMPT_TEMPLATE,
    ONTOLOGY_MERGE_PROMPT_TEMPLATE,
)

from .retrieval_prompts import (
    RETRIEVAL_PLAN_PROMPT,
    COMPRESSION_PROMPT,
    OBSERVATION_COMPRESSION_PROMPT,
    PATTERN_TO_CYPHER_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RETRIEVAL_QUALITY_PROMPT,
    format_retrieval_plan_prompt,
    format_compression_prompt,
    format_observation_compression_prompt,
    format_pattern_to_cypher_prompt,
)

__all__ = [
    # KG extraction prompts
    "KG_EXTRACTION_PROMPT",
    "KG_EXTRACTION_PROMPT_MINIMAL",
    "KG_EXTRACTION_PROMPT_COMPACT",
    "DEFAULT_EXTRACTION_PROMPT",
    "get_extraction_prompt",
    "ONTOLOGY_MAPPING_PROMPT_TEMPLATE",
    "FROZEN_ONTOLOGY_MAPPING_PROMPT_TEMPLATE",
    "LABEL_GENERALIZATION_PROMPT_TEMPLATE",
    "ENTITY_VERIFICATION_PROMPT",
    "LABEL_VERIFICATION_PROMPT_TEMPLATE",
    "ONTOLOGY_MERGE_PROMPT_TEMPLATE",
    # Retrieval prompts
    "RETRIEVAL_PLAN_PROMPT",
    "COMPRESSION_PROMPT",
    "OBSERVATION_COMPRESSION_PROMPT",
    "PATTERN_TO_CYPHER_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
    "RETRIEVAL_QUALITY_PROMPT",
    "format_retrieval_plan_prompt",
    "format_compression_prompt",
    "format_observation_compression_prompt",
    "format_pattern_to_cypher_prompt",
]
