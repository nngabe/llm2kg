"""
Prompts for CLaRa-style retrieval planning and context compression.

These prompts support agentic retrieval with:
1. Retrieval planning - creating explicit plans for what to retrieve
2. Context compression - CLaRa-style compression of retrieved context to relevant portions
3. Pattern-to-Cypher conversion for relationship queries
"""

# =============================================================================
# RETRIEVAL PLANNING PROMPT
# =============================================================================

RETRIEVAL_PLAN_PROMPT = """You are a retrieval planning expert. Given a question, create an explicit plan for what information to retrieve from a knowledge graph and other sources.

Question: {question}

Think step-by-step about what information you need:
1. What entities are explicitly mentioned or implied?
2. What relationships between entities matter for answering this question?
3. What facts need to be verified or looked up?
4. What might require external search if the knowledge graph doesn't have it?

Create a retrieval plan as JSON:
{{
    "reasoning": "Brief explanation of why this plan will help answer the question",
    "information_needs": [
        "Specific piece of information needed #1",
        "Specific piece of information needed #2"
    ],
    "entity_targets": [
        "Entity Name 1",
        "Entity Name 2"
    ],
    "relationship_queries": [
        "Entity1 -[RELATIONSHIP_TYPE]-> ?",
        "? -[CAUSES]-> Entity2"
    ],
    "fallback_searches": [
        "Web search query if graph doesn't have the answer"
    ]
}}

Guidelines:
- entity_targets: Use exact entity names from the question or commonly known canonical names
- relationship_queries: Use patterns like "Entity -[REL_TYPE]-> ?" or "? -[REL_TYPE]-> Entity"
  - Common relationship types: IS_A, PART_OF, CAUSES, INFLUENCES, ENABLES, PRODUCES, USES, LOCATED_AT
- fallback_searches: Simple web search queries for external information
- Keep the plan focused - typically 2-5 entity targets and 2-4 relationship queries

Return ONLY valid JSON, no explanation."""


# =============================================================================
# CONTEXT COMPRESSION PROMPT (CLaRa-style)
# =============================================================================

COMPRESSION_PROMPT = """You are a context compression expert. Given a question and retrieved context, extract ONLY the facts relevant to answering the question.

Question: {question}

Retrieved Context:
{context}

Instructions:
1. Identify facts directly relevant to answering the question
2. Remove redundant or irrelevant information
3. Preserve entity names, relationships, and key attributes exactly
4. Keep causal chains and logical connections intact
5. Format as concise bullet points

Compressed context (max 500 tokens):"""


# =============================================================================
# OBSERVATION COMPRESSION PROMPT
# =============================================================================

OBSERVATION_COMPRESSION_PROMPT = """Compress this tool observation to only the facts relevant to the question.

Question: {question}

Tool Observation:
{observation}

Extract key facts as bullet points (max 200 tokens):"""


# =============================================================================
# PATTERN TO CYPHER PROMPT
# =============================================================================

PATTERN_TO_CYPHER_PROMPT = """Convert this natural language relationship pattern to a Cypher query.

Pattern: {pattern}

The pattern uses this notation:
- "Entity -[REL_TYPE]-> ?" means: find what Entity connects to via REL_TYPE
- "? -[REL_TYPE]-> Entity" means: find what connects to Entity via REL_TYPE
- "Entity1 -[?]-> Entity2" means: find relationship between Entity1 and Entity2
- "Entity -[*1..2]-> ?" means: traverse 1-2 hops from Entity

Return ONLY the Cypher query, no explanation.

Example patterns and their Cypher:
- "Apple -[PRODUCES]-> ?" -> MATCH (a {{name: 'Apple'}})-[r:PRODUCES]->(b) RETURN a, r, b
- "? -[CAUSES]-> Climate Change" -> MATCH (a)-[r:CAUSES]->(b {{name: 'Climate Change'}}) RETURN a, r, b
- "Einstein -[?]-> ?" -> MATCH (a {{name: 'Einstein'}})-[r]->(b) RETURN a, type(r) as rel_type, b LIMIT 20

Cypher query:"""


# =============================================================================
# ENTITY EXTRACTION FROM QUESTION
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract entity names from this question that should be looked up in a knowledge graph.

Question: {question}

Return a JSON list of entity names mentioned or implied:
["Entity1", "Entity2", ...]

Guidelines:
- Include proper nouns (people, places, organizations, products)
- Include technical terms or concepts that might be in the knowledge graph
- Use canonical names (e.g., "United States" not "US")
- Don't include generic words like "cause", "effect", "relationship"

Return ONLY the JSON list."""


# =============================================================================
# RETRIEVAL QUALITY CHECK PROMPT
# =============================================================================

RETRIEVAL_QUALITY_PROMPT = """Evaluate if the retrieved context is sufficient to answer the question.

Question: {question}

Retrieved Context:
{context}

Evaluation:
1. Can the question be fully answered with this context? (yes/no)
2. What information is missing?
3. Should we search for more entities or use web search?

Return JSON:
{{
    "sufficient": true/false,
    "missing_info": ["missing item 1", "missing item 2"],
    "suggested_actions": [
        {{"action": "graph_lookup", "target": "Entity Name"}},
        {{"action": "web_search", "query": "search query"}}
    ]
}}"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_retrieval_plan_prompt(question: str) -> str:
    """Format the retrieval plan prompt with the question."""
    return RETRIEVAL_PLAN_PROMPT.format(question=question)


def format_compression_prompt(question: str, context: str, max_context_chars: int = 8000) -> str:
    """Format the compression prompt with question and context."""
    # Truncate context if too long
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n... [truncated]"
    return COMPRESSION_PROMPT.format(question=question, context=context)


def format_observation_compression_prompt(question: str, observation: str) -> str:
    """Format the observation compression prompt."""
    return OBSERVATION_COMPRESSION_PROMPT.format(question=question, observation=observation)


def format_pattern_to_cypher_prompt(pattern: str) -> str:
    """Format the pattern-to-cypher prompt."""
    return PATTERN_TO_CYPHER_PROMPT.format(pattern=pattern)
