#!/usr/bin/env python3
"""
Generate golden test cases using GPT-5.2 for the Agent Evaluation Suite.

This script:
1. Queries Neo4j for existing entities/relationships
2. Uses GPT-5.2 to generate comprehensive test cases across all layers
3. Saves to golden dataset format
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/benchmarks")

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def get_entity_sample(driver, limit: int = 30) -> List[Dict[str, Any]]:
    """Get sample entities from Neo4j for test case generation."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND size(e.description) > 30
            RETURN e.name as name, e.ontology_type as type,
                   substring(e.description, 0, 200) as description
            ORDER BY e.extraction_count DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(record) for record in result]


def get_relationship_sample(driver, limit: int = 30) -> List[Dict[str, Any]]:
    """Get sample relationships from Neo4j for test case generation."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.description IS NOT NULL AND b.description IS NOT NULL
            RETURN a.name as source, type(r) as rel_type, b.name as target,
                   a.ontology_type as source_type, b.ontology_type as target_type
            LIMIT $limit
        """, limit=limit)
        return [dict(record) for record in result]


def generate_test_cases_with_gpt52(entities: List[Dict], relationships: List[Dict]) -> List[Dict]:
    """Use GPT-5.2 to generate comprehensive test cases across all layers."""

    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.7,
    )

    # Format entity and relationship info for the prompt
    entity_info = "\n".join([
        f"- {e['name']} ({e['type']}): {e['description'][:100]}..."
        for e in entities[:15]
    ])

    rel_info = "\n".join([
        f"- {r['source']} --[{r['rel_type']}]--> {r['target']}"
        for r in relationships[:15]
    ])

    prompt = f"""Generate a comprehensive set of test cases for evaluating a GraphRAG + ReAct QA agent.

The knowledge graph contains entities like:
{entity_info}

And relationships like:
{rel_info}

Generate exactly 20 test cases across 4 evaluation layers:

1. RETRIEVAL (5 cases) - Questions that test graph lookup and context retrieval
   - Must reference entities that exist in the graph
   - Include expected_entities that should be retrieved

2. AGENTIC (5 cases) - Questions requiring multi-hop reasoning and tool use
   - Test tool selection (graph_lookup, web_search, cypher_query)
   - Include optimal_tool_sequence for evaluation
   - Include 2 rejection tests (questions that SHOULD be rejected: real-time data, personal info, future predictions)

3. INTEGRITY (5 cases) - Questions testing schema adherence and entity disambiguation
   - Test if agent correctly handles new entity creation
   - Test if agent avoids duplicate entities
   - Include should_reject=true for invalid entities

4. GENERATION (5 cases) - Questions testing answer quality and faithfulness
   - Test comprehensive answer generation
   - Include ground_truth_context for faithfulness evaluation
   - Include expected_answer for comparison

Return as a JSON array with this format for each case:
{{
    "id": "layer_type_001",
    "question": "The question text",
    "expected_answer": "Expected answer (null for rejection/agentic)",
    "expected_entities": ["Entity1", "Entity2"],
    "expected_relationships": [{{"source": "A", "type": "REL", "target": "B"}}],
    "ground_truth_context": ["Context sentence 1", "Context sentence 2"],
    "optimal_tool_sequence": ["graph_lookup", "web_search"],
    "minimum_steps": 2,
    "should_reject": false,
    "rejection_reason": null,
    "type": "retrieval|agentic|integrity|generation",
    "difficulty": "easy|medium|hard",
    "source": "gpt52_generated",
    "reviewed": false,
    "metadata": {{"note": "optional notes"}}
}}

Return ONLY the JSON array, no markdown formatting."""

    logger.info("Calling GPT-5.2 to generate test cases...")
    response = llm.invoke(prompt)

    # Parse response
    try:
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        test_cases = json.loads(content)
        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT-5.2 response: {e}")
        logger.error(f"Raw response: {response.content[:500]}")
        return []


def generate_manual_test_cases(entities: List[Dict], relationships: List[Dict]) -> List[Dict]:
    """Generate manual test cases if GPT-5.2 is unavailable."""
    test_cases = []

    # RETRIEVAL LAYER (5 cases)
    for i, entity in enumerate(entities[:3]):
        test_cases.append({
            "id": f"retrieval_entity_{i+1:03d}",
            "question": f"What is {entity['name']} and what is its role?",
            "expected_answer": entity['description'][:200],
            "expected_entities": [entity['name']],
            "expected_relationships": [],
            "ground_truth_context": [entity['description']],
            "optimal_tool_sequence": ["graph_lookup"],
            "minimum_steps": 2,
            "should_reject": False,
            "type": "retrieval",
            "difficulty": "easy",
            "source": "manual_generated",
            "reviewed": False,
        })

    for i, rel in enumerate(relationships[:2]):
        test_cases.append({
            "id": f"retrieval_rel_{i+1:03d}",
            "question": f"What is the relationship between {rel['source']} and {rel['target']}?",
            "expected_entities": [rel['source'], rel['target']],
            "expected_relationships": [{"source": rel['source'], "type": rel['rel_type'], "target": rel['target']}],
            "ground_truth_context": [],
            "optimal_tool_sequence": ["graph_lookup", "cypher_query"],
            "minimum_steps": 3,
            "should_reject": False,
            "type": "retrieval",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
        })

    # AGENTIC LAYER (5 cases)
    test_cases.extend([
        {
            "id": "agentic_multihop_001",
            "question": f"How does {entities[0]['name']} affect {entities[1]['name']} through intermediate concepts?",
            "expected_entities": [entities[0]['name'], entities[1]['name']],
            "expected_relationships": [],
            "optimal_tool_sequence": ["graph_lookup", "cypher_query", "graph_lookup"],
            "minimum_steps": 4,
            "should_reject": False,
            "type": "agentic",
            "difficulty": "hard",
            "source": "manual_generated",
            "reviewed": False,
        },
        {
            "id": "agentic_tool_selection_001",
            "question": f"Compare {entities[2]['name']} with recent developments in the field.",
            "expected_entities": [entities[2]['name']],
            "optimal_tool_sequence": ["graph_lookup", "web_search"],
            "minimum_steps": 3,
            "should_reject": False,
            "type": "agentic",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
        },
        {
            "id": "agentic_rejection_001",
            "question": "What is the current price of Bitcoin in USD?",
            "expected_entities": [],
            "should_reject": True,
            "rejection_reason": "Real-time financial data not available in knowledge graph",
            "type": "agentic",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"category": "real_time_data"},
        },
        {
            "id": "agentic_rejection_002",
            "question": "What will be the GDP of the United States in 2030?",
            "expected_entities": [],
            "should_reject": True,
            "rejection_reason": "Future predictions cannot be made from knowledge graph",
            "type": "agentic",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"category": "future_prediction"},
        },
        {
            "id": "agentic_cypher_001",
            "question": f"Find all entities related to {entities[3]['name']} within 2 hops.",
            "expected_entities": [entities[3]['name']],
            "optimal_tool_sequence": ["cypher_query"],
            "minimum_steps": 2,
            "should_reject": False,
            "type": "agentic",
            "difficulty": "hard",
            "source": "manual_generated",
            "reviewed": False,
        },
    ])

    # INTEGRITY LAYER (5 cases)
    test_cases.extend([
        {
            "id": "integrity_schema_001",
            "question": f"Add information about a new concept called 'Quantum Economics' that relates to {entities[0]['name']}.",
            "expected_entities": ["Quantum Economics", entities[0]['name']],
            "should_reject": False,
            "type": "integrity",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests schema adherence for new entity creation"},
        },
        {
            "id": "integrity_disambiguation_001",
            "question": f"Is '{entities[0]['name']}' the same as '{entities[0]['name'].upper()}'?",
            "expected_entities": [entities[0]['name']],
            "should_reject": False,
            "type": "integrity",
            "difficulty": "easy",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests entity disambiguation"},
        },
        {
            "id": "integrity_duplicate_001",
            "question": f"Create an entry for '{entities[1]['name']}' with new information.",
            "expected_entities": [entities[1]['name']],
            "should_reject": True,
            "rejection_reason": "Entity already exists, should merge not duplicate",
            "type": "integrity",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests duplicate detection"},
        },
        {
            "id": "integrity_source_001",
            "question": "Tell me about a fictional company called Acme Quantum Labs founded in 2075.",
            "expected_entities": ["Acme Quantum Labs"],
            "should_reject": True,
            "rejection_reason": "Fictional/fabricated entities should not be added",
            "type": "integrity",
            "difficulty": "hard",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests source citation requirements"},
        },
        {
            "id": "integrity_relationship_001",
            "question": f"What type of relationship would best describe the connection between {entities[4]['name']} and {entities[5]['name']}?",
            "expected_entities": [entities[4]['name'], entities[5]['name']],
            "should_reject": False,
            "type": "integrity",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests relationship type assignment"},
        },
    ])

    # GENERATION LAYER (5 cases)
    test_cases.extend([
        {
            "id": "generation_comprehensive_001",
            "question": f"Provide a detailed explanation of {entities[0]['name']} including its components and applications.",
            "expected_answer": entities[0]['description'],
            "expected_entities": [entities[0]['name']],
            "ground_truth_context": [entities[0]['description']],
            "optimal_tool_sequence": ["graph_lookup"],
            "minimum_steps": 2,
            "should_reject": False,
            "type": "generation",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
        },
        {
            "id": "generation_synthesis_001",
            "question": f"Synthesize information about {entities[0]['name']} and {entities[1]['name']} to explain their connection.",
            "expected_entities": [entities[0]['name'], entities[1]['name']],
            "ground_truth_context": [entities[0]['description'], entities[1]['description']],
            "optimal_tool_sequence": ["graph_lookup", "graph_lookup"],
            "minimum_steps": 3,
            "should_reject": False,
            "type": "generation",
            "difficulty": "hard",
            "source": "manual_generated",
            "reviewed": False,
        },
        {
            "id": "generation_faithfulness_001",
            "question": f"What are the key characteristics of {entities[2]['name']}?",
            "expected_answer": entities[2]['description'],
            "expected_entities": [entities[2]['name']],
            "ground_truth_context": [entities[2]['description']],
            "optimal_tool_sequence": ["graph_lookup"],
            "minimum_steps": 2,
            "should_reject": False,
            "type": "generation",
            "difficulty": "easy",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests answer faithfulness to source"},
        },
        {
            "id": "generation_citation_001",
            "question": f"Explain {entities[3]['name']} with proper source attribution.",
            "expected_entities": [entities[3]['name']],
            "ground_truth_context": [entities[3]['description']],
            "should_reject": False,
            "type": "generation",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests citation accuracy"},
        },
        {
            "id": "generation_relevance_001",
            "question": f"Why is {entities[4]['name']} important?",
            "expected_entities": [entities[4]['name']],
            "ground_truth_context": [entities[4]['description']],
            "should_reject": False,
            "type": "generation",
            "difficulty": "medium",
            "source": "manual_generated",
            "reviewed": False,
            "metadata": {"note": "Tests answer relevance"},
        },
    ])

    return test_cases


def save_golden_dataset(test_cases: List[Dict], output_path: str):
    """Save test cases as golden dataset."""
    dataset = {
        "name": "comprehensive_v1",
        "version": "1.0",
        "description": "Comprehensive golden dataset covering all 4 evaluation layers",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {
            "author": "Agent Eval Suite",
            "purpose": "Complete evaluation matrix testing",
            "layers": ["retrieval", "agentic", "integrity", "generation"],
        },
        "test_cases": test_cases,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Saved {len(test_cases)} test cases to {output_path}")
    return output_path


def main():
    logger.info("=" * 60)
    logger.info("GENERATING COMPREHENSIVE GOLDEN DATASET")
    logger.info("=" * 60)

    # Connect to Neo4j
    logger.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Get sample data
    entities = get_entity_sample(driver, limit=30)
    relationships = get_relationship_sample(driver, limit=30)

    logger.info(f"Retrieved {len(entities)} entities and {len(relationships)} relationships")

    # Try GPT-5.2 first, fall back to manual generation
    try:
        test_cases = generate_test_cases_with_gpt52(entities, relationships)
        if not test_cases:
            raise ValueError("GPT-5.2 returned empty test cases")
    except Exception as e:
        logger.warning(f"GPT-5.2 generation failed: {e}")
        logger.info("Falling back to manual test case generation...")
        test_cases = generate_manual_test_cases(entities, relationships)

    # Print summary by layer
    layer_counts = {}
    for tc in test_cases:
        layer = tc.get("type", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    logger.info("\nTest case distribution by layer:")
    for layer, count in sorted(layer_counts.items()):
        logger.info(f"  {layer}: {count}")

    # Save dataset
    output_path = "benchmarks/agent_eval/golden/datasets/comprehensive_v1.json"
    save_golden_dataset(test_cases, output_path)

    driver.close()
    return test_cases


if __name__ == "__main__":
    main()
