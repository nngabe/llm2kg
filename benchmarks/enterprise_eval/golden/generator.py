"""
Golden Dataset Generator.

LLM-based generation of test cases from graph structure.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schema import GoldenTestCase, GoldenDataset

logger = logging.getLogger(__name__)


GENERATION_PROMPT = """You are generating test cases for evaluating a GraphRAG + ReAct agent.

The agent has access to a knowledge graph with the following structure:

Entities in the graph:
{entities}

Relationships in the graph:
{relationships}

Generate a {test_type} test case with difficulty: {difficulty}

Requirements for {test_type} test cases:
{type_requirements}

The test case should be {difficulty} difficulty:
- easy: Single entity lookup, direct relationships
- medium: Multi-hop reasoning, 2-3 entities involved
- hard: Complex reasoning, multiple relationships, potential ambiguity

Return a JSON object with this structure:
{{
    "question": "The question to ask the agent",
    "expected_answer": "The correct answer based on the graph",
    "expected_entities": ["list", "of", "entities", "needed"],
    "expected_relationships": [
        {{"source": "Entity1", "type": "REL_TYPE", "target": "Entity2"}}
    ],
    "ground_truth_context": ["relevant context items"],
    "optimal_tool_sequence": ["tool1", "tool2", ...],
    "minimum_steps": 3,
    "should_reject": false,
    "rejection_reason": null,
    "reasoning": "Why this is a good test case"
}}
"""

TYPE_REQUIREMENTS = {
    "retrieval": """
- Focus on testing context retrieval quality
- Question should require finding specific entities/relationships
- Ground truth context should be clearly defined
- Test both precision (finding relevant info) and recall (finding all relevant info)
""",
    "agentic": """
- Focus on testing agent decision-making
- Question should require multiple tool calls
- Include an optimal tool sequence
- Test tool selection and argument correctness
""",
    "integrity": """
- Focus on testing graph update quality
- Question should potentially trigger new entity/relationship creation
- Include expected schema-compliant types
- Test disambiguation and schema adherence
""",
    "generation": """
- Focus on testing answer generation quality
- Question should have a clear, verifiable answer
- Include expected citations/sources
- Test faithfulness and relevance
""",
    "rejection": """
- Create an UNANSWERABLE question
- Question should be outside graph knowledge
- Set should_reject: true
- Include rejection_reason explaining why it's unanswerable
- The agent should recognize it cannot answer
""",
}


class GoldenDatasetGenerator:
    """Generates golden test cases using LLM."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self._llm = None
        self._neo4j_driver = None

        # Neo4j config
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def _get_llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return self._llm

    def _get_neo4j_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._neo4j_driver is None and self._neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    self._neo4j_uri,
                    auth=(self._neo4j_user, self._neo4j_password),
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
        return self._neo4j_driver

    def _get_graph_structure(self) -> Dict[str, Any]:
        """Get entities and relationships from Neo4j."""
        driver = self._get_neo4j_driver()

        if not driver:
            # Return sample structure if no Neo4j connection
            return {
                "entities": [
                    {"name": "Sample Entity 1", "type": "Organization"},
                    {"name": "Sample Entity 2", "type": "Person"},
                ],
                "relationships": [
                    {"source": "Sample Entity 2", "type": "WORKS_FOR", "target": "Sample Entity 1"}
                ],
            }

        try:
            with driver.session() as session:
                # Get sample entities
                entities_result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.name as name, labels(n)[0] as type, n.description as description
                    LIMIT 50
                """)
                entities = [
                    {"name": r["name"], "type": r["type"], "description": r["description"]}
                    for r in entities_result
                ]

                # Get sample relationships
                rels_result = session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a.name as source, type(r) as type, b.name as target
                    LIMIT 50
                """)
                relationships = [
                    {"source": r["source"], "type": r["type"], "target": r["target"]}
                    for r in rels_result
                ]

                return {"entities": entities, "relationships": relationships}

        except Exception as e:
            logger.error(f"Failed to query Neo4j: {e}")
            return {"entities": [], "relationships": []}

    def _generate_single(
        self,
        test_type: str,
        difficulty: str,
        graph_structure: Dict[str, Any],
    ) -> Optional[GoldenTestCase]:
        """Generate a single test case.

        Args:
            test_type: Type of test case
            difficulty: Difficulty level
            graph_structure: Graph entities and relationships

        Returns:
            Generated test case or None on failure
        """
        from langchain_core.messages import HumanMessage

        prompt = GENERATION_PROMPT.format(
            entities=json.dumps(graph_structure["entities"][:20], indent=2),
            relationships=json.dumps(graph_structure["relationships"][:20], indent=2),
            test_type=test_type,
            difficulty=difficulty,
            type_requirements=TYPE_REQUIREMENTS.get(test_type, ""),
        )

        try:
            llm = self._get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                data = json.loads(content[start_idx:end_idx])

                # Create test case
                test_case = GoldenTestCase(
                    id=f"{test_type}_{difficulty}_{uuid.uuid4().hex[:8]}",
                    question=data.get("question", ""),
                    expected_answer=data.get("expected_answer"),
                    expected_entities=data.get("expected_entities", []),
                    expected_relationships=data.get("expected_relationships", []),
                    ground_truth_context=data.get("ground_truth_context", []),
                    optimal_tool_sequence=data.get("optimal_tool_sequence", []),
                    minimum_steps=data.get("minimum_steps"),
                    should_reject=data.get("should_reject", False),
                    rejection_reason=data.get("rejection_reason"),
                    type=test_type,
                    difficulty=difficulty,
                    source="generated",
                    generated_by=self.model,
                    generated_at=datetime.now().isoformat(),
                    generation_prompt=prompt[:500],  # Store truncated prompt
                    metadata={"reasoning": data.get("reasoning", "")},
                )

                return test_case

            logger.warning(f"Could not parse JSON from response: {content[:200]}")
            return None

        except Exception as e:
            logger.error(f"Failed to generate test case: {e}")
            return None

    def generate_dataset(
        self,
        name: str,
        counts: Dict[str, Dict[str, int]] = None,
    ) -> GoldenDataset:
        """Generate a complete golden dataset.

        Args:
            name: Dataset name
            counts: Dict of {test_type: {difficulty: count}}
                   Default generates 2 of each type/difficulty combo

        Returns:
            Generated GoldenDataset
        """
        if counts is None:
            counts = {
                "retrieval": {"easy": 2, "medium": 2, "hard": 1},
                "agentic": {"easy": 2, "medium": 2, "hard": 1},
                "integrity": {"easy": 1, "medium": 2, "hard": 1},
                "generation": {"easy": 2, "medium": 2, "hard": 1},
                "rejection": {"easy": 1, "medium": 2, "hard": 1},
            }

        dataset = GoldenDataset(
            name=name,
            description=f"Auto-generated dataset using {self.model}",
        )

        # Get graph structure once
        graph_structure = self._get_graph_structure()
        logger.info(
            f"Loaded graph structure: {len(graph_structure['entities'])} entities, "
            f"{len(graph_structure['relationships'])} relationships"
        )

        total = sum(sum(d.values()) for d in counts.values())
        generated = 0

        for test_type, difficulties in counts.items():
            for difficulty, count in difficulties.items():
                logger.info(f"Generating {count} {difficulty} {test_type} test cases...")

                for i in range(count):
                    test_case = self._generate_single(
                        test_type=test_type,
                        difficulty=difficulty,
                        graph_structure=graph_structure,
                    )

                    if test_case:
                        dataset.add_test_case(test_case)
                        generated += 1
                        logger.info(f"Generated {generated}/{total}: {test_case.id}")
                    else:
                        logger.warning(f"Failed to generate {test_type}/{difficulty} #{i+1}")

        logger.info(f"Generated {generated}/{total} test cases")
        return dataset

    def close(self):
        """Clean up resources."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
