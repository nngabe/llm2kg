"""
System prompts for the agent_skb.py knowledge graph extraction agent.

All prompts are defined here for maintainability and easy modification.
Template prompts (ending with _TEMPLATE) require formatting with runtime values.

IMPORTANT: All curly braces in JSON examples must be escaped as {{ and }}
because these prompts are used with LangChain's ChatPromptTemplate.
"""

# =============================================================================
# PREFERRED RELATIONSHIP TYPES (Shared across prompts)
# =============================================================================

PREFERRED_RELATIONSHIP_TYPES = """
Preferred relationship types (use when applicable):
IS_A, PART_OF, HAS_PART, RELATES_TO, INFLUENCES, CAUSES, CAUSES_EFFECT_ON,
CREATES, PRODUCES, LOCATED_IN, CONTAINS, MEMBER_OF, WORKS_FOR, OWNS,
USES, IMPLEMENTS, DEPENDS_ON, SUPPORTS, OPPOSES, PRECEDES, FOLLOWS
"""

# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION - MINIMAL (for very short texts < 500 chars)
# =============================================================================

KG_EXTRACTION_PROMPT_MINIMAL = """Extract entities and relationships as a knowledge graph.

Node format: {{"id": "Name", "labels": ["Type"], "properties": {{"description": "Brief description"}}}}
Relationship format: {{"source": "id1", "target": "id2", "type": "RELATION_TYPE", "properties": {{"description": "Context"}}}}

Use PascalCase for labels, SCREAMING_SNAKE_CASE for relationship types.
Return valid JSON with "nodes" and "relationships" arrays."""


# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION - COMPACT (for short texts 500-1200 chars)
# =============================================================================

KG_EXTRACTION_PROMPT_COMPACT = """You are a knowledge graph extraction expert.

RULES:
- Node IDs: Use canonical names from the text (not pronouns like "it" or "the company")
- Node Labels: PascalCase (Person, Organization, Concept, Event, Location, Process)
- Relationship Types: SCREAMING_SNAKE_CASE
""" + PREFERRED_RELATIONSHIP_TYPES + """

OUTPUT: Valid JSON only.
{{
  "nodes": [{{"id": "Entity Name", "labels": ["Type"], "properties": {{"description": "10-15 word summary"}}}}],
  "relationships": [{{"source": "id1", "target": "id2", "type": "RELATION_TYPE", "properties": {{"description": "Context"}}}}]
}}

EXAMPLE:
Input: "Apple Inc., founded by Steve Jobs, manufactures the iPhone."
Output:
{{
  "nodes": [
    {{"id": "Apple Inc.", "labels": ["Organization", "Company"], "properties": {{"description": "Technology company that manufactures consumer electronics including the iPhone."}}}},
    {{"id": "Steve Jobs", "labels": ["Person"], "properties": {{"description": "Co-founder of Apple Inc. who played a key role in the company's establishment."}}}},
    {{"id": "iPhone", "labels": ["Product"], "properties": {{"description": "Smartphone product manufactured by Apple Inc."}}}}
  ],
  "relationships": [
    {{"source": "Steve Jobs", "target": "Apple Inc.", "type": "FOUNDED", "properties": {{"description": "Steve Jobs co-founded Apple Inc."}}}},
    {{"source": "Apple Inc.", "target": "iPhone", "type": "PRODUCES", "properties": {{"description": "Apple manufactures the iPhone product line."}}}}
  ]
}}"""


# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION - STANDARD (default, for texts 1200-3000 chars)
# =============================================================================

KG_EXTRACTION_PROMPT = """You are a Neo4j Knowledge Graph Extraction expert. Your goal is to extract structured data from text for import into a graph database.

### NAMING CONVENTIONS (STRICT)
1. **Node Labels:** Use PascalCase (e.g., `Organization`, `EconomicSystem`, `PublicCompany`).
2. **Relationship Types:** Use SCREAMING_SNAKE_CASE. Prefer these canonical types when applicable:
   IS_A, PART_OF, HAS_PART, RELATES_TO, INFLUENCES, CAUSES, CREATES, PRODUCES,
   LOCATED_IN, CONTAINS, MEMBER_OF, WORKS_FOR, OWNS, USES, IMPLEMENTS,
   DEPENDS_ON, SUPPORTS, OPPOSES, CHARACTERIZED_BY, CONTROLS, BENEFITS
3. **Entity IDs:** Use the most distinct, canonical name found in the text (e.g., use "Apple" instead of "The company").

### EXTRACTION LOGIC
1. **Resolve Coreferences:** If the text says "The company" or "It" referring to a previous entity, use the specific entity name for the ID.
2. **Split Lists:** "Alice and Bob like Pizza" becomes TWO relationships: (Alice)-[:LIKES]->(Pizza) and (Bob)-[:LIKES]->(Pizza).
3. **Detailed Descriptions:** Both node and edge descriptions must be detailed sentences (10-20 words) summarizing the context found in the text.
4. **Ignore Noise:** Do not extract figures, citations (e.g., "[12]"), or meta-commentary.

### OUTPUT SCHEMA
Return ONLY valid JSON.
{{
  "nodes": [
    {{
      "id": "string (unique identifier)",
      "labels": ["string"],
      "properties": {{
        "description": "string (detailed summary)"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "string (must match a node id)",
      "target": "string (must match a node id)",
      "type": "string (SCREAMING_SNAKE_CASE)",
      "properties": {{
        "description": "string (detailed explanation of the link)"
      }}
    }}
  ]
}}

### EXAMPLES

#### Example 1: Lists & Distributive Relationships
**Input:** "While copper is abundant, both gold and silver are rare precious metals found deep in the Earth's crust. These materials are highly valued in electronics manufacturing and jewelry making due to their superior conductivity and aesthetic appeal."

**Output:**
{{
  "nodes": [
    {{"id": "Gold", "labels": ["Material"], "properties": {{"description": "A rare precious metal found deep in the Earth's crust, valued for its conductivity and aesthetics."}}}},
    {{"id": "Silver", "labels": ["Material"], "properties": {{"description": "A rare precious metal found alongside gold, known for high conductivity and use in jewelry."}}}},
    {{"id": "Precious Metal", "labels": ["Concept"], "properties": {{"description": "A category of rare and economically valuable metals found in the Earth's crust."}}}},
    {{"id": "Electronics Manufacturing", "labels": ["Industry"], "properties": {{"description": "An industrial sector that utilizes conductive metals for building components."}}}},
    {{"id": "Jewelry Making", "labels": ["Industry"], "properties": {{"description": "A craft industry that utilizes precious metals for their aesthetic appeal."}}}}
  ],
  "relationships": [
    {{"source": "Gold", "target": "Precious Metal", "type": "IS_A", "properties": {{"description": "Gold is categorized as a precious metal due to its rarity and value."}}}},
    {{"source": "Silver", "target": "Precious Metal", "type": "IS_A", "properties": {{"description": "Silver is categorized as a precious metal due to its rarity and value."}}}},
    {{"source": "Gold", "target": "Electronics Manufacturing", "type": "USED_IN", "properties": {{"description": "Gold is used in electronics manufacturing because of its superior conductivity."}}}},
    {{"source": "Silver", "target": "Jewelry Making", "type": "USED_IN", "properties": {{"description": "Silver is utilized in the creation of jewelry due to its aesthetic qualities."}}}}
  ]
}}

#### Example 2: Noise, Synonyms & Appositives
**Input:** "On a Tuesday morning, SpaceX, a private aerospace firm, launched the massive Starship rocket from their base in Texas. Unfortunately, the experimental vehicle (Fig 2.1) exploded shortly after liftoff due to a pressure valve failure, halting operations temporarily."

**Output:**
{{
  "nodes": [
    {{"id": "SpaceX", "labels": ["Organization", "Company"], "properties": {{"description": "A private aerospace firm responsible for launching the Starship rocket."}}}},
    {{"id": "Starship", "labels": ["Artifact", "Rocket"], "properties": {{"description": "A massive experimental rocket developed and launched by SpaceX."}}}},
    {{"id": "Texas", "labels": ["Location"], "properties": {{"description": "The geographic location where SpaceX operates its launch base."}}}},
    {{"id": "Pressure Valve Failure", "labels": ["Event", "Failure"], "properties": {{"description": "A mechanical malfunction that caused the explosion of the rocket."}}}}
  ],
  "relationships": [
    {{"source": "SpaceX", "target": "Starship", "type": "LAUNCHED", "properties": {{"description": "SpaceX conducted the launch of the Starship rocket from their Texas base."}}}},
    {{"source": "Starship", "target": "Texas", "type": "LAUNCHED_FROM", "properties": {{"description": "The rocket began its flight trajectory from a launchpad in Texas."}}}},
    {{"source": "Pressure Valve Failure", "target": "Starship", "type": "CAUSED_DESTRUCTION_OF", "properties": {{"description": "A failure in the pressure valve led to the explosion of the vehicle."}}}}
  ]
}}

#### Example 3: Passive Voice & Complex States
**Input:** "In a rigid command economy, production prices are strictly set by the government to ensure basic affordability. Conversely, modern market economies rely on the fluctuating forces of supply and demand to determine the value of goods and services."

**Output:**
{{
  "nodes": [
    {{"id": "Command Economy", "labels": ["EconomicSystem"], "properties": {{"description": "A rigid economic system where the government controls pricing to ensure affordability."}}}},
    {{"id": "Government", "labels": ["Authority"], "properties": {{"description": "The central ruling body that strictly sets prices in a command economy."}}}},
    {{"id": "Production Prices", "labels": ["Concept"], "properties": {{"description": "The monetary value assigned to goods, which is controlled centrally in some systems."}}}},
    {{"id": "Market Economy", "labels": ["EconomicSystem"], "properties": {{"description": "A modern economic system that determines value through fluctuating market forces."}}}},
    {{"id": "Supply And Demand", "labels": ["Process"], "properties": {{"description": "The fluctuating market forces that determine the value of goods in a market economy."}}}}
  ],
  "relationships": [
    {{"source": "Government", "target": "Production Prices", "type": "CONTROLS", "properties": {{"description": "The government actively sets and controls production prices to maintain affordability."}}}},
    {{"source": "Command Economy", "target": "Government", "type": "CHARACTERIZED_BY", "properties": {{"description": "This economic system is defined by the central role of the government in decision making."}}}},
    {{"source": "Market Economy", "target": "Supply And Demand", "type": "RELIES_ON", "properties": {{"description": "Market economies depend on the forces of supply and demand to function effectively."}}}}
  ]
}}
"""


# =============================================================================
# ONTOLOGY MAPPING (Dynamic Label Type)
# =============================================================================

ONTOLOGY_MAPPING_PROMPT_TEMPLATE = """You are an ontology expert. Map the extracted {label_type} to an appropriate canonical ontology label.

Current ontology {label_type}s: {current_ontology}
Target: {target_min}-{target_max} canonical {label_type}s that abstractly represent all extracted types.

Rules:
1. If an existing ontology label is a good match, use it exactly
2. If no good match exists, propose a new abstract label that could cover this and similar types
3. Prefer broader, reusable categories over specific ones
4. Return ONLY the ontology label, nothing else"""


# =============================================================================
# FROZEN ONTOLOGY MAPPING (When Ontology is Locked)
# =============================================================================

FROZEN_ONTOLOGY_MAPPING_PROMPT_TEMPLATE = """The ontology is frozen. Map the extracted {label_type} to the BEST matching existing ontology label.
You MUST choose from the existing labels only.

Existing ontology {label_type}s: {ontology}

Return ONLY the chosen ontology label, nothing else."""


# =============================================================================
# LABEL GENERALIZATION (Creating Broader Types)
# =============================================================================

LABEL_GENERALIZATION_PROMPT_TEMPLATE = """You are an ontology expert. Convert this specific {label_type} into a broader, more abstract canonical {label_type}.

The goal is to have {target_min}-{target_max} canonical {label_type}s total that can cover all possible extractions.

Examples for entity types: Person, Organization, Concept, Event, Location, Work, Process
Examples for relationship types: RELATES_TO, INFLUENCES, CREATES, PART_OF, LOCATED_IN, OCCURS_IN

Return ONLY the generalized label, nothing else."""


# =============================================================================
# ENTITY VERIFICATION (Checking if Two Entities are the Same)
# =============================================================================

ENTITY_VERIFICATION_PROMPT = """You are a data cleaning expert. Determine if two entities refer to the same real-world object.
Consider both the names and descriptions when making your decision."""


# =============================================================================
# LABEL VERIFICATION (Checking if Two Labels are the Same)
# =============================================================================

LABEL_VERIFICATION_PROMPT_TEMPLATE = """You are an ontology expert. Determine if two {category} labels refer to the same concept.
If they are the same, pick the more standard/canonical label."""


# =============================================================================
# ONTOLOGY MERGE PROPOSAL (Consolidating Types)
# =============================================================================

ONTOLOGY_MERGE_PROMPT_TEMPLATE = """You are an ontology expert. The current ontology has too many {label_type}.
Propose merges to consolidate similar or related types into broader canonical types.

Target: {target_min}-{target_max} {label_type}

Rules:
1. Merge specific types into more abstract parent types
2. Keep the most general/reusable types
3. Return a JSON object mapping merge_from -> merge_to
4. Only propose merges, don't remove types without merging
5. Return ONLY valid JSON, no explanation

Example output: {{"Economist": "Person", "Scientist": "Person", "developed": "CREATES"}}"""


# =============================================================================
# ADAPTIVE PROMPT SELECTION
# =============================================================================

def get_extraction_prompt(text_length: int) -> str:
    """
    Select the appropriate extraction prompt based on input text length.

    Args:
        text_length: Character count of the input text

    Returns:
        The most appropriate extraction prompt for the text length

    Rationale:
        - Short texts (<600 chars): Use minimal prompt to avoid overwhelming the model
        - Medium texts (600-1500 chars): Use compact prompt with one example
        - Long texts (>1500 chars): Use full prompt with multiple examples
    """
    if text_length < 600:
        return KG_EXTRACTION_PROMPT_MINIMAL
    elif text_length < 1500:
        return KG_EXTRACTION_PROMPT_COMPACT
    else:
        return KG_EXTRACTION_PROMPT
