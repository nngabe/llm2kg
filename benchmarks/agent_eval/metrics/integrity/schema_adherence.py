"""
Schema Adherence Metric.

Deterministic metric checking if graph updates conform to the ontology schema.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class SchemaAdherenceMetric(BaseMetric):
    """Measures schema adherence of graph updates.

    Schema Adherence = valid_items / total_items

    This metric deterministically checks:
    - Entity types match ontology
    - Relationship types match ontology
    - Required properties are present
    - Property values are valid
    """

    layer = EvaluationLayer.INTEGRITY
    name = "Schema Adherence"

    def __init__(self, threshold: float = 0.95):
        super().__init__(threshold)
        self._valid_entity_types: Optional[Set[str]] = None
        self._valid_relationship_types: Optional[Set[str]] = None
        self._ontology_loaded = False

    def _load_ontology(self, ontology: Optional[Any] = None) -> bool:
        """Load valid types from ontology.

        Args:
            ontology: Ontology instance

        Returns:
            True if ontology loaded successfully
        """
        if self._ontology_loaded:
            return True

        if ontology:
            try:
                # Try to get entity labels from ontology
                if hasattr(ontology, 'get_entity_labels'):
                    self._valid_entity_types = set(ontology.get_entity_labels())
                elif hasattr(ontology, 'entity_types'):
                    self._valid_entity_types = set(ontology.entity_types)
                elif hasattr(ontology, 'ENTITY_TYPES'):
                    self._valid_entity_types = set(ontology.ENTITY_TYPES)

                # Try to get relationship labels from ontology
                if hasattr(ontology, 'get_relationship_labels'):
                    self._valid_relationship_types = set(ontology.get_relationship_labels())
                elif hasattr(ontology, 'relationship_types'):
                    self._valid_relationship_types = set(ontology.relationship_types)
                elif hasattr(ontology, 'RELATIONSHIP_TYPES'):
                    self._valid_relationship_types = set(ontology.RELATIONSHIP_TYPES)

                self._ontology_loaded = True
                logger.info(
                    f"Loaded ontology: {len(self._valid_entity_types or [])} entity types, "
                    f"{len(self._valid_relationship_types or [])} relationship types"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load ontology: {e}")

        # Fallback: common defaults
        self._valid_entity_types = {
            "Agent", "Organization", "Event", "Concept", "Location",
            "Technology", "Product", "Person", "Document", "Policy",
            "Entity",  # Generic fallback
        }
        self._valid_relationship_types = {
            "CAUSES", "INFLUENCES", "RELATES_TO", "PART_OF", "HAS",
            "WORKS_FOR", "LOCATED_IN", "USES", "CREATED", "MENTIONS",
            "LEADS_TO", "ASSOCIATED_WITH", "DEPENDS_ON", "AFFECTS",
        }
        logger.info("Using default ontology types")
        return False

    def _validate_entity(
        self,
        entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a single entity against the schema.

        Args:
            entity: Entity dictionary with type, name, properties

        Returns:
            Validation result dictionary
        """
        violations = []
        warnings = []

        # Extract entity type
        entity_type = entity.get("ontology_type") or entity.get("type") or entity.get("label")

        if not entity_type:
            violations.append("Missing entity type")
        elif self._valid_entity_types and entity_type not in self._valid_entity_types:
            violations.append(f"Invalid entity type: {entity_type}")

        # Check required properties
        name = entity.get("name") or entity.get("id")
        if not name:
            violations.append("Missing entity name/id")

        # Check for empty properties (warning, not violation)
        properties = entity.get("properties", {})
        for key, value in properties.items():
            if value is None or value == "":
                warnings.append(f"Empty property: {key}")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "entity_type": entity_type,
            "entity_name": name,
        }

    def _validate_relationship(
        self,
        relationship: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a single relationship against the schema.

        Args:
            relationship: Relationship dictionary with type, source, target

        Returns:
            Validation result dictionary
        """
        violations = []
        warnings = []

        # Extract relationship type
        rel_type = relationship.get("type") or relationship.get("relationship_type")

        if not rel_type:
            violations.append("Missing relationship type")
        elif self._valid_relationship_types and rel_type not in self._valid_relationship_types:
            violations.append(f"Invalid relationship type: {rel_type}")

        # Check source and target
        source = relationship.get("source") or relationship.get("from") or relationship.get("source_id")
        target = relationship.get("target") or relationship.get("to") or relationship.get("target_id")

        if not source:
            violations.append("Missing relationship source")
        if not target:
            violations.append("Missing relationship target")

        # Check for self-loops (warning)
        if source and target and source == target:
            warnings.append("Self-referential relationship")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "relationship_type": rel_type,
            "source": source,
            "target": target,
        }

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        ontology: Optional[Any] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure schema adherence.

        Args:
            test_case: Test case
            agent_output: Agent output with new nodes/edges
            ontology: Optional ontology instance

        Returns:
            MetricResult with schema adherence score
        """
        # Load ontology
        self._load_ontology(ontology)

        new_nodes = agent_output.new_nodes
        new_edges = agent_output.new_edges

        if not new_nodes and not new_edges:
            return self._create_result(
                score=1.0,
                details={"note": "No graph updates to validate"},
            )

        # Validate entities
        entity_results = []
        valid_entities = 0
        for entity in new_nodes:
            result = self._validate_entity(entity)
            entity_results.append(result)
            if result["valid"]:
                valid_entities += 1

        # Validate relationships
        relationship_results = []
        valid_relationships = 0
        for relationship in new_edges:
            result = self._validate_relationship(relationship)
            relationship_results.append(result)
            if result["valid"]:
                valid_relationships += 1

        # Calculate overall score
        total_items = len(new_nodes) + len(new_edges)
        valid_items = valid_entities + valid_relationships
        score = valid_items / total_items if total_items > 0 else 1.0

        # Collect all violations
        all_violations = []
        for result in entity_results + relationship_results:
            all_violations.extend(result.get("violations", []))

        all_warnings = []
        for result in entity_results + relationship_results:
            all_warnings.extend(result.get("warnings", []))

        return self._create_result(
            score=score,
            details={
                "total_entities": len(new_nodes),
                "valid_entities": valid_entities,
                "total_relationships": len(new_edges),
                "valid_relationships": valid_relationships,
                "total_items": total_items,
                "valid_items": valid_items,
                "violations": all_violations,
                "warnings": all_warnings,
                "entity_validations": entity_results,
                "relationship_validations": relationship_results,
                "valid_entity_types": list(self._valid_entity_types or [])[:10],
                "valid_relationship_types": list(self._valid_relationship_types or [])[:10],
            },
        )
