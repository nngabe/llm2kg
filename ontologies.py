"""
Ontology configurations for Knowledge Graph extraction.

Provides three pre-defined ontology sizes:
- SMALL: 8 entity types, 12 relationship types (fast processing)
- MEDIUM: 12 entity types, 16 relationship types (balanced)
- LARGE: 15 entity types, 20 relationship types (academic/research)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class OntologyConfig:
    """Configuration for a knowledge graph ontology."""
    name: str
    description: str
    entity_types: Dict[str, str]  # label -> description
    relationship_types: Dict[str, str]  # label -> description

    def get_entity_labels(self) -> List[str]:
        """Get list of entity type labels."""
        return list(self.entity_types.keys())

    def get_relationship_labels(self) -> List[str]:
        """Get list of relationship type labels."""
        return list(self.relationship_types.keys())

    def format_entity_types_for_prompt(self) -> str:
        """Format entity types with descriptions for inclusion in prompts."""
        lines = ["ENTITY TYPES (use these labels for node types):"]
        for label, desc in self.entity_types.items():
            lines.append(f"  - {label}: {desc}")
        return "\n".join(lines)

    def format_relationship_types_for_prompt(self) -> str:
        """Format relationship types with descriptions for inclusion in prompts."""
        lines = ["RELATIONSHIP TYPES (use these labels for edge types):"]
        for label, desc in self.relationship_types.items():
            lines.append(f"  - {label}: {desc}")
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Format complete ontology for inclusion in extraction prompts."""
        return f"""### STRICT ONTOLOGY
You MUST use ONLY these labels. Do not invent new types.

{self.format_entity_types_for_prompt()}

{self.format_relationship_types_for_prompt()}
"""


# =============================================================================
# SMALL ONTOLOGY (8 entity types, 12 relationship types)
# =============================================================================
ONTOLOGY_SMALL = OntologyConfig(
    name="small",
    description="Minimal ontology for fast processing and simple domains",
    entity_types={
        "Agent": "Individuals or autonomous systems (e.g., Marie Curie, ChatGPT)",
        "Organization": "Formal institutions or companies (e.g., NASA, Google)",
        "PhysicalObject": "Tangible items, artifacts, or substances (e.g., H2O, microscope)",
        "Event": "Discrete happenings with start/end (e.g., WWII, Big Bang)",
        "Process": "Continuous workflows or cycles (e.g., Photosynthesis, Peer Review)",
        "Location": "Geographic or spatial regions (e.g., Paris, Mars)",
        "Concept": "General ideas, topics, or categories (e.g., Democracy, Entropy)",
        "Quantity": "Measurable values with units (e.g., 100 meters, $5 billion)",
    },
    relationship_types={
        "IS_A": "Classification (e.g., Gold IS_A Metal)",
        "PART_OF": "Composition (e.g., Wheel PART_OF Car)",
        "CAUSES": "Direct causality (e.g., Smoking CAUSES Cancer)",
        "INFLUENCES": "Weaker causal link (e.g., Weather INFLUENCES Mood)",
        "ENABLES": "Prerequisite (e.g., Internet ENABLES E-commerce)",
        "PRECEDES": "Happens before (e.g., Thunder PRECEDES Lightning)",
        "LOCATED_AT": "Physical position",
        "PARTICIPATES_IN": "Agent involvement (e.g., US PARTICIPATES_IN NATO)",
        "PRODUCES": "Output generation (e.g., Factory PRODUCES Cars)",
        "USES": "Instrumental utility (e.g., Painter USES Brush)",
        "SUPPORTS": "Evidence backing a claim",
        "CONTRADICTS": "Evidence opposing a claim",
    },
)


# =============================================================================
# MEDIUM ONTOLOGY (12 entity types, 16 relationship types)
# =============================================================================
ONTOLOGY_MEDIUM = OntologyConfig(
    name="medium",
    description="Balanced ontology for most domains",
    entity_types={
        "Agent": "Individuals or autonomous systems (e.g., Marie Curie, ChatGPT)",
        "Organization": "Formal institutions or companies (e.g., NASA, Google)",
        "Group": "Informal collections of people/things (e.g., middle class, protons)",
        "PhysicalObject": "Tangible items, artifacts, or substances (e.g., H2O, microscope)",
        "Resource": "Assets used in processes (money, energy, data)",
        "Event": "Discrete happenings with start/end (e.g., WWII, Big Bang)",
        "Process": "Continuous workflows or cycles (e.g., Photosynthesis, Peer Review)",
        "Location": "Geographic or spatial regions (e.g., Paris, Mars)",
        "TimePeriod": "Specific eras, dates, or intervals (e.g., Renaissance, Q1 2024)",
        "Concept": "General ideas, topics, or categories (e.g., Democracy, Entropy)",
        "Attribute": "Qualities or characteristics (e.g., Volatility, Intelligence)",
        "Quantity": "Measurable values with units (e.g., 100 meters, $5 billion)",
    },
    relationship_types={
        "IS_A": "Classification (e.g., Gold IS_A Metal)",
        "PART_OF": "Composition (e.g., Wheel PART_OF Car)",
        "CONTAINS": "Inverse of PART_OF (e.g., Cell CONTAINS Nucleus)",
        "CAUSES": "Direct causality (e.g., Smoking CAUSES Cancer)",
        "INFLUENCES": "Weaker causal link (e.g., Weather INFLUENCES Mood)",
        "ENABLES": "Prerequisite (e.g., Internet ENABLES E-commerce)",
        "PREVENTS": "Negative causality (e.g., Vaccine PREVENTS Disease)",
        "PRECEDES": "Happens before (e.g., Thunder PRECEDES Lightning)",
        "DURING": "Happens within a time (e.g., Battle DURING WWII)",
        "LOCATED_AT": "Physical position",
        "PARTICIPATES_IN": "Agent involvement (e.g., US PARTICIPATES_IN NATO)",
        "PERFORMS": "Executing a process (e.g., Doctor PERFORMS Surgery)",
        "PRODUCES": "Output generation (e.g., Factory PRODUCES Cars)",
        "USES": "Instrumental utility (e.g., Painter USES Brush)",
        "SUPPORTS": "Evidence backing a claim",
        "CONTRADICTS": "Evidence opposing a claim",
    },
)


# =============================================================================
# LARGE ONTOLOGY (15 entity types, 20 relationship types)
# =============================================================================
ONTOLOGY_LARGE = OntologyConfig(
    name="large",
    description="Full ontology for academic/research with maximum expressiveness",
    entity_types={
        # Actors
        "Agent": "Individuals or autonomous systems (e.g., Marie Curie, ChatGPT, The President)",
        "Organization": "Formal institutions or companies (e.g., NASA, University of Oxford, Google)",
        "Group": "Informal collections of people/things (e.g., middle class, baby boomers, protons)",
        # Objects
        "PhysicalObject": "Tangible items, artifacts, or substances (e.g., Electron microscope, H2O)",
        "Resource": "Assets used in processes (e.g., Funding, 50kWh electricity, Census Data)",
        # Actions
        "Event": "Discrete happenings with start/end (e.g., The Big Bang, WWII, 2008 Financial Crisis)",
        "Process": "Continuous workflows or biological/chemical cycles (e.g., Photosynthesis, Peer Review)",
        # Space/Time
        "Location": "Geographic or spatial regions (e.g., Paris, Mars, The Prefrontal Cortex)",
        "TimePeriod": "Specific eras, dates, or intervals (e.g., The Renaissance, Q1 2024, Late Jurassic)",
        # Abstract
        "Concept": "General ideas, topics, or categories (e.g., Democracy, Entropy, Feminism)",
        "Methodology": "Techniques, algorithms, or approaches (e.g., Double-blind study, Gradient Descent, PCR)",
        "Proposition": "Specific assertions, theorems, or laws (e.g., Newton's 2nd Law, Riemann Hypothesis)",
        # Properties
        "Attribute": "Qualities or characteristics (e.g., Volatility, Intelligence, Solubility)",
        "Quantity": "Measurable values with units (e.g., 100 meters, 98.6 degrees, $5 billion)",
        "Regulation": "Rules, standards, or governing laws (e.g., GDPR, The Constitution, ISO 9001)",
    },
    relationship_types={
        # Hierarchical
        "IS_A": "Classification (e.g., Gold IS_A Metal)",
        "PART_OF": "Composition (e.g., Wheel PART_OF Car)",
        "CONTAINS": "Inverse of PART_OF (e.g., Cell CONTAINS Nucleus)",
        # Causal
        "CAUSES": "Direct causality (e.g., Smoking CAUSES Cancer)",
        "INFLUENCES": "Weaker causal link (e.g., Weather INFLUENCES Mood)",
        "ENABLES": "Prerequisite (e.g., Internet ENABLES E-commerce)",
        "PREVENTS": "Negative causality (e.g., Vaccine PREVENTS Disease)",
        # Temporal
        "PRECEDES": "Happens before (e.g., Thunder PRECEDES Lightning)",
        "DURING": "Happens within a time (e.g., Battle DURING WWII)",
        # Spatial
        "LOCATED_AT": "Physical position",
        "ORIGINATES_FROM": "Source or provenance (e.g., Data ORIGINATES_FROM Survey)",
        # Action
        "PARTICIPATES_IN": "Agent involvement (e.g., US PARTICIPATES_IN NATO)",
        "PERFORMS": "Executing a process (e.g., Doctor PERFORMS Surgery)",
        "USES": "Instrumental utility (e.g., Painter USES Brush)",
        "PRODUCES": "Output generation (e.g., Factory PRODUCES Cars)",
        # Logic/Argument
        "SUPPORTS": "Evidence backing a claim",
        "CONTRADICTS": "Evidence opposing a claim",
        "PROVES": "Logical certainty (e.g., Proof PROVES Theorem)",
        # Measurement
        "HAS_VALUE": "Linking object to Quantity (e.g., Rod HAS_VALUE 5kg)",
        # Meta
        "DEFINED_BY": "Citations/References (e.g., Term DEFINED_BY Paper X)",
    },
)


# =============================================================================
# ONTOLOGY REGISTRY
# =============================================================================
ONTOLOGIES = {
    "small": ONTOLOGY_SMALL,
    "medium": ONTOLOGY_MEDIUM,
    "large": ONTOLOGY_LARGE,
}

DEFAULT_ONTOLOGY = "medium"


def get_ontology(name: str = None) -> OntologyConfig:
    """
    Get an ontology configuration by name.

    Args:
        name: Ontology name ('small', 'medium', 'large'). Defaults to 'medium'.

    Returns:
        OntologyConfig for the specified ontology.

    Raises:
        ValueError: If ontology name is not recognized.
    """
    name = name or DEFAULT_ONTOLOGY
    if name not in ONTOLOGIES:
        raise ValueError(f"Unknown ontology: {name}. Choose from: {list(ONTOLOGIES.keys())}")
    return ONTOLOGIES[name]


def list_ontologies() -> List[Tuple[str, str, int, int]]:
    """
    List all available ontologies with their stats.

    Returns:
        List of (name, description, entity_count, relationship_count) tuples.
    """
    return [
        (name, config.description, len(config.entity_types), len(config.relationship_types))
        for name, config in ONTOLOGIES.items()
    ]
