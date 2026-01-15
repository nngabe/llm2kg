"""
Entity Disambiguation Metric.

Vector similarity + LLM metric detecting potential duplicate entities.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import LLMJudgeMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


DISAMBIGUATION_PROMPT = """You are checking if a newly created entity is a duplicate of an existing entity.

New Entity:
- Name: {new_name}
- Type: {new_type}
- Properties: {new_properties}

Potential Duplicate (similarity score: {similarity:.2f}):
- Name: {existing_name}
- Type: {existing_type}
- Properties: {existing_properties}

Determine if these represent the same real-world entity.

Consider:
- Are the names variations of the same thing? (e.g., "Microsoft" vs "Microsoft Corporation")
- Do they refer to the same person/organization/concept?
- Could they reasonably be merged?

Return a JSON object:
{{
    "is_duplicate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "merge_recommendation": "which entity should be kept, or 'keep_both' if not duplicates"
}}
"""


class EntityDisambiguationMetric(LLMJudgeMetric):
    """Measures entity disambiguation quality.

    Entity Disambiguation = 1.0 - (duplicates_created * penalty)

    This metric checks if newly created entities are duplicates of existing ones
    using vector similarity search and LLM verification.
    """

    layer = EvaluationLayer.INTEGRITY
    name = "Entity Disambiguation"

    def __init__(
        self,
        threshold: float = 0.90,
        similarity_threshold: float = 0.92,
        duplicate_penalty: float = 0.2,
        judge_provider: str = "google",
        judge_model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        fallback_provider: str = "openai",
        fallback_model: str = "gpt-5.2",
    ):
        super().__init__(
            threshold=threshold,
            judge_provider=judge_provider,
            judge_model=judge_model,
            temperature=temperature,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )
        self.similarity_threshold = similarity_threshold
        self.duplicate_penalty = duplicate_penalty

    def _find_similar_entities_neo4j(
        self,
        entity: Dict[str, Any],
        neo4j_driver: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar entities using Neo4j vector search.

        Args:
            entity: New entity to check
            neo4j_driver: Neo4j driver
            top_k: Number of similar entities to retrieve

        Returns:
            List of similar entities with scores
        """
        query = """
        MATCH (n:Entity)
        WHERE n.name <> $new_name
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        WHERE score > $threshold
        RETURN node.name as name, node.type as type,
               properties(node) as properties, score
        ORDER BY score DESC
        """

        try:
            with neo4j_driver.session() as session:
                # Get embedding for new entity
                embedding = entity.get("embedding")
                if not embedding:
                    # Try to get from properties
                    embedding = entity.get("properties", {}).get("embedding")

                if not embedding:
                    logger.warning("No embedding for entity, skipping vector search")
                    return []

                result = session.run(
                    query,
                    new_name=entity.get("name", ""),
                    embedding=embedding,
                    top_k=top_k,
                    threshold=self.similarity_threshold,
                )

                return [
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "properties": record["properties"],
                        "similarity": record["score"],
                    }
                    for record in result
                ]
        except Exception as e:
            logger.warning(f"Neo4j vector search failed: {e}")
            return []

    def _find_similar_entities_fallback(
        self,
        entity: Dict[str, Any],
        existing_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fallback: find similar entities by name matching.

        Args:
            entity: New entity to check
            existing_entities: List of existing entities

        Returns:
            List of similar entities
        """
        new_name = (entity.get("name") or "").lower().strip()
        if not new_name:
            return []

        similar = []
        for existing in existing_entities:
            existing_name = (existing.get("name") or "").lower().strip()
            if not existing_name:
                continue

            # Check various similarity conditions
            similarity = 0.0

            # Exact match (shouldn't happen but check anyway)
            if new_name == existing_name:
                similarity = 1.0
            # One is substring of other
            elif new_name in existing_name or existing_name in new_name:
                similarity = 0.85
            # Common tokens
            else:
                new_tokens = set(new_name.split())
                existing_tokens = set(existing_name.split())
                if new_tokens and existing_tokens:
                    overlap = len(new_tokens & existing_tokens)
                    total = len(new_tokens | existing_tokens)
                    similarity = overlap / total

            if similarity >= 0.5:  # Lower threshold for fallback
                similar.append({
                    "name": existing.get("name"),
                    "type": existing.get("type"),
                    "properties": existing.get("properties", {}),
                    "similarity": similarity,
                })

        # Sort by similarity
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:5]

    def _check_duplicate_llm(
        self,
        new_entity: Dict[str, Any],
        existing_entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to verify if entities are duplicates.

        Args:
            new_entity: New entity
            existing_entity: Potentially duplicate entity

        Returns:
            Verification result
        """
        prompt = DISAMBIGUATION_PROMPT.format(
            new_name=new_entity.get("name", "Unknown"),
            new_type=new_entity.get("type", "Unknown"),
            new_properties=str(new_entity.get("properties", {}))[:500],
            existing_name=existing_entity.get("name", "Unknown"),
            existing_type=existing_entity.get("type", "Unknown"),
            existing_properties=str(existing_entity.get("properties", {}))[:500],
            similarity=existing_entity.get("similarity", 0.0),
        )

        result = self._invoke_judge(prompt)

        if "error" in result:
            # On error, be conservative and assume not duplicate
            return {
                "is_duplicate": False,
                "confidence": 0.0,
                "reasoning": f"Judge error: {result['error']}",
                "merge_recommendation": "keep_both",
            }

        return result

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        neo4j_driver: Optional[Any] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure entity disambiguation quality.

        Args:
            test_case: Test case
            agent_output: Agent output with new nodes
            neo4j_driver: Optional Neo4j driver for vector search

        Returns:
            MetricResult with disambiguation score
        """
        new_nodes = agent_output.new_nodes

        if not new_nodes:
            return self._create_result(
                score=1.0,
                details={"note": "No new entities to check for duplicates"},
            )

        # Check each new entity for duplicates
        duplicate_checks = []
        confirmed_duplicates = 0

        for entity in new_nodes:
            # Find similar entities
            if neo4j_driver:
                similar = self._find_similar_entities_neo4j(entity, neo4j_driver)
            else:
                # Use existing entities from context as fallback
                existing = [
                    item for item in agent_output.context_items
                    if item.get("source_type") in ("entity", "graph")
                ]
                similar = self._find_similar_entities_fallback(entity, existing)

            entity_check = {
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
                "similar_entities": [],
                "is_duplicate": False,
            }

            for sim_entity in similar:
                # Verify with LLM
                verification = self._check_duplicate_llm(entity, sim_entity)

                check_result = {
                    "similar_to": sim_entity.get("name"),
                    "similarity": sim_entity.get("similarity"),
                    **verification,
                }
                entity_check["similar_entities"].append(check_result)

                if verification.get("is_duplicate", False):
                    entity_check["is_duplicate"] = True
                    confirmed_duplicates += 1
                    break  # One duplicate is enough

            duplicate_checks.append(entity_check)

        # Calculate score
        # Each duplicate reduces score by penalty
        penalty = confirmed_duplicates * self.duplicate_penalty
        score = max(0.0, 1.0 - penalty)

        return self._create_result(
            score=score,
            details={
                "total_new_entities": len(new_nodes),
                "confirmed_duplicates": confirmed_duplicates,
                "penalty_per_duplicate": self.duplicate_penalty,
                "similarity_threshold": self.similarity_threshold,
                "duplicate_checks": duplicate_checks,
                "method": "neo4j_vector" if neo4j_driver else "fallback",
            },
        )
