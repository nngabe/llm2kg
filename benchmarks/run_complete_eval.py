#!/usr/bin/env python3
"""
Complete Evaluation Script for Enterprise Eval Suite.

Runs 2 test cases per layer (8 total) to complete the full evaluation matrix.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/benchmarks")

from neo4j import GraphDatabase
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def get_sample_entities(driver, limit: int = 10) -> List[Dict[str, Any]]:
    """Get sample entities from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND size(e.description) > 50
            RETURN e.name as name, e.ontology_type as type,
                   substring(e.description, 0, 300) as description
            ORDER BY e.extraction_count DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(record) for record in result]


def get_sample_relationships(driver, limit: int = 10) -> List[Dict[str, Any]]:
    """Get sample relationships from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.description IS NOT NULL AND b.description IS NOT NULL
            RETURN a.name as source, type(r) as rel_type, b.name as target,
                   a.ontology_type as source_type, b.ontology_type as target_type
            LIMIT $limit
        """, limit=limit)
        return [dict(record) for record in result]


def create_test_cases_all_layers(entities: List[Dict], relationships: List[Dict]) -> List[Dict[str, Any]]:
    """Create 2 test cases per layer for comprehensive evaluation."""
    test_cases = []

    # === RETRIEVAL LAYER (2 cases) ===
    # Test 1: Simple entity lookup
    test_cases.append({
        "id": "retrieval_001",
        "question": f"What is {entities[0]['name']}?",
        "expected_entities": [entities[0]['name']],
        "expected_answer": entities[0]['description'],
        "ground_truth_context": [entities[0]['description']],
        "optimal_tool_sequence": ["graph_lookup"],
        "minimum_steps": 2,
        "type": "retrieval",
        "difficulty": "easy",
        "should_reject": False,
    })

    # Test 2: Relationship lookup
    test_cases.append({
        "id": "retrieval_002",
        "question": f"What is the relationship between {relationships[0]['source']} and {relationships[0]['target']}?",
        "expected_entities": [relationships[0]['source'], relationships[0]['target']],
        "expected_relationships": [{"source": relationships[0]['source'], "type": relationships[0]['rel_type'], "target": relationships[0]['target']}],
        "optimal_tool_sequence": ["graph_lookup", "cypher_query"],
        "minimum_steps": 3,
        "type": "retrieval",
        "difficulty": "medium",
        "should_reject": False,
    })

    # === AGENTIC LAYER (2 cases) ===
    # Test 1: Multi-hop reasoning
    test_cases.append({
        "id": "agentic_001",
        "question": f"How does {entities[1]['name']} influence economic outcomes?",
        "expected_entities": [entities[1]['name']],
        "optimal_tool_sequence": ["graph_lookup", "cypher_query"],
        "minimum_steps": 3,
        "type": "agentic",
        "difficulty": "medium",
        "should_reject": False,
    })

    # Test 2: Rejection test (real-time data)
    test_cases.append({
        "id": "agentic_002",
        "question": "What is the current stock price of Apple Inc.?",
        "expected_entities": [],
        "type": "agentic",
        "difficulty": "medium",
        "should_reject": True,
        "rejection_reason": "Real-time stock data not available in knowledge graph",
        "metadata": {"category": "real_time_data"},
    })

    # === INTEGRITY LAYER (2 cases) ===
    # Test 1: Entity disambiguation
    test_cases.append({
        "id": "integrity_001",
        "question": f"Is '{entities[2]['name']}' the same entity as '{entities[2]['name'].lower()}'?",
        "expected_entities": [entities[2]['name']],
        "type": "integrity",
        "difficulty": "easy",
        "should_reject": False,
        "metadata": {"note": "Tests entity disambiguation - should recognize same entity"},
    })

    # Test 2: Duplicate detection
    test_cases.append({
        "id": "integrity_002",
        "question": f"Tell me about a fictional company called 'QuantumAI Labs' that was founded in 2099.",
        "expected_entities": ["QuantumAI Labs"],
        "type": "integrity",
        "difficulty": "hard",
        "should_reject": True,
        "rejection_reason": "Fictional/fabricated entities should not be created without valid sources",
        "metadata": {"note": "Tests source citation requirements"},
    })

    # === GENERATION LAYER (2 cases) ===
    # Test 1: Comprehensive answer
    test_cases.append({
        "id": "generation_001",
        "question": f"Provide a detailed explanation of {entities[3]['name']}.",
        "expected_answer": entities[3]['description'],
        "expected_entities": [entities[3]['name']],
        "ground_truth_context": [entities[3]['description']],
        "optimal_tool_sequence": ["graph_lookup"],
        "minimum_steps": 2,
        "type": "generation",
        "difficulty": "medium",
        "should_reject": False,
    })

    # Test 2: Synthesis from multiple sources
    test_cases.append({
        "id": "generation_002",
        "question": f"What are the key characteristics of {entities[4]['name']}?",
        "expected_answer": entities[4]['description'],
        "expected_entities": [entities[4]['name']],
        "ground_truth_context": [entities[4]['description']],
        "optimal_tool_sequence": ["graph_lookup"],
        "minimum_steps": 2,
        "type": "generation",
        "difficulty": "easy",
        "should_reject": False,
        "metadata": {"note": "Tests answer faithfulness to source"},
    })

    return test_cases


def _extract_step_data(step) -> Dict[str, Any]:
    """Extract thought step data from either dict or ThoughtStep object."""
    if isinstance(step, dict):
        thought = step.get("thought", "")
        action = step.get("action")
        observation = step.get("observation", "")
        if action:
            if isinstance(action, dict):
                action_data = {"tool_name": action.get("tool_name", ""), "arguments": action.get("arguments", {})}
            else:
                action_data = {"tool_name": getattr(action, 'tool_name', ''), "arguments": getattr(action, 'arguments', {})}
        else:
            action_data = None
    else:
        thought = getattr(step, 'thought', '')
        observation = getattr(step, 'observation', '')
        action = getattr(step, 'action', None)
        if action:
            action_data = {"tool_name": getattr(action, 'tool_name', ''), "arguments": getattr(action, 'arguments', {})}
        else:
            action_data = None
    return {"thought": thought, "action": action_data, "observation": observation}


def _extract_context_items(response) -> List[Dict[str, Any]]:
    """Extract context items from reasoning steps."""
    context_items = []
    if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
        for step in response.reasoning_steps:
            step_data = _extract_step_data(step)
            action = step_data.get("action")
            observation = step_data.get("observation", "")
            if action and action.get("tool_name") == "graph_lookup":
                entity_name = action.get("arguments", {}).get("entity_name", "")
                if entity_name:
                    context_items.append({
                        "source_type": "entity",
                        "source_id": entity_name,
                        "content": observation,
                        "relevance_score": 1.0,
                        "metadata": {"entity_name": entity_name},
                    })
            elif action and action.get("tool_name") == "cypher_query":
                query = action.get("arguments", {}).get("query", "")
                context_items.append({
                    "source_type": "neo4j",
                    "source_id": "cypher_query",
                    "content": observation,
                    "relevance_score": 1.0,
                    "metadata": {"query": query},
                })
    if hasattr(response, 'citations') and response.citations:
        for c in response.citations:
            source_type = getattr(c, 'source_type', '') if hasattr(c, 'source_type') else c.get('source_type', '')
            if source_type == "graph":
                source_id = getattr(c, 'source_id', '') if hasattr(c, 'source_id') else c.get('source_id', '')
                context_items.append({
                    "source_type": "entity",
                    "source_id": source_id,
                    "content": getattr(c, 'excerpt', '') if hasattr(c, 'excerpt') else c.get('excerpt', ''),
                    "relevance_score": 1.0,
                    "metadata": {"entity_name": source_id},
                })
    return context_items


def run_qa_agent(questions: List[Dict[str, Any]]) -> List[Tuple[Dict, Any]]:
    """Run QA agent on questions."""
    from agent_qa import ReActQAAgent

    logger.info("Initializing QA Agent...")
    agent = ReActQAAgent()

    results = []
    for i, q in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {q['question'][:50]}...")
        try:
            response = agent.answer_question(q['question'])
            results.append((q, response))
            logger.info(f"  Answer: {response.answer[:80]}...")
            logger.info(f"  Confidence: {response.confidence:.2f}")
            logger.info(f"  Citations: {len(response.citations)}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append((q, None))

    agent.close()
    return results


def convert_to_test_cases(questions: List[Dict[str, Any]]):
    """Convert questions to TestCase format."""
    from enterprise_eval.metrics.base import TestCase

    test_cases = []
    for q in questions:
        tc = TestCase(
            id=q['id'],
            question=q['question'],
            expected_answer=q.get('expected_answer'),
            expected_entities=q.get('expected_entities', []),
            expected_relationships=q.get('expected_relationships', []),
            ground_truth_context=q.get('ground_truth_context', []),
            optimal_tool_sequence=q.get('optimal_tool_sequence', []),
            minimum_steps=q.get('minimum_steps'),
            type=q.get('type', 'all'),
            difficulty=q.get('difficulty', 'medium'),
            should_reject=q.get('should_reject', False),
            rejection_reason=q.get('rejection_reason'),
            metadata=q.get('metadata', {}),
        )
        test_cases.append(tc)
    return test_cases


def convert_to_agent_outputs(results: List[Tuple[Dict, Any]]):
    """Convert QA responses to AgentOutput format."""
    from enterprise_eval.metrics.base import AgentOutput

    outputs = []
    for q, response in results:
        if response is None:
            output = AgentOutput(
                question=q['question'],
                answer="ERROR: Agent failed",
                confidence=0.0,
            )
        else:
            citations = []
            if hasattr(response, 'citations') and response.citations:
                for c in response.citations:
                    if isinstance(c, dict):
                        citations.append(c)
                    else:
                        citations.append({
                            "source_type": c.source_type,
                            "source_id": c.source_id,
                            "source_title": getattr(c, 'source_title', None),
                            "trust_level": c.trust_level,
                            "excerpt": c.excerpt,
                        })

            thought_history = []
            if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
                for step in response.reasoning_steps:
                    thought_history.append(_extract_step_data(step))

            context_items = _extract_context_items(response)

            output = AgentOutput(
                question=q['question'],
                answer=response.answer,
                confidence=response.confidence,
                citations=citations,
                thought_history=thought_history,
                context_items=context_items,
                external_info_used=response.external_info_used,
            )
        outputs.append(output)
    return outputs


def run_evaluation(test_cases, agent_outputs):
    """Run the evaluation suite."""
    from enterprise_eval import EvalConfig
    from enterprise_eval.runner import create_default_runner
    from enterprise_eval.reporting import HumanReporter, JSONReporter

    logger.info("Running evaluation suite...")

    config = EvalConfig()
    runner = create_default_runner(config)

    result = runner.run_evaluation(test_cases, agent_outputs)

    # Print human-readable report
    reporter = HumanReporter(use_colors=True)
    reporter.print_report(result)

    # Save JSON report
    json_reporter = JSONReporter("benchmarks/enterprise_eval/reports")
    report_path = json_reporter.generate(result, f"complete_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info(f"JSON report saved to {report_path}")

    runner.close()
    return result


def main():
    logger.info("=" * 60)
    logger.info("COMPLETE 4-LAYER EVALUATION MATRIX")
    logger.info("=" * 60)

    # Connect to Neo4j
    logger.info("\n--- Step 1: Querying Neo4j for sample data ---")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    entities = get_sample_entities(driver, limit=10)
    relationships = get_sample_relationships(driver, limit=10)

    logger.info(f"Found {len(entities)} sample entities")
    logger.info(f"Found {len(relationships)} sample relationships")

    # Create test cases for all 4 layers
    logger.info("\n--- Step 2: Creating test cases (2 per layer) ---")
    all_questions = create_test_cases_all_layers(entities, relationships)

    # Print test cases by layer
    layer_counts = {}
    for q in all_questions:
        layer = q['type']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print("\n=== TEST CASES BY LAYER ===")
    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer}: {count} test cases")
        for q in all_questions:
            if q['type'] == layer:
                reject = " [REJECTION]" if q.get('should_reject') else ""
                print(f"    - [{q['id']}] {q['question'][:50]}...{reject}")

    # Run QA agent
    logger.info("\n--- Step 3: Running QA Agent ---")
    results = run_qa_agent(all_questions)

    # Print results summary
    print("\n=== QA AGENT RESULTS ===")
    for q, response in results:
        status = "OK" if response else "FAILED"
        if response:
            conf = f"{response.confidence:.2f}"
            answer_preview = response.answer[:60].replace('\n', ' ')
            external = "web" if response.external_info_used else "graph"
        else:
            conf = "N/A"
            answer_preview = "ERROR"
            external = "N/A"
        print(f"  [{status}] {q['id']} ({q['type']}): conf={conf}, source={external}")
        print(f"       Q: {q['question'][:50]}...")
        print(f"       A: {answer_preview}...")

    # Run evaluation
    logger.info("\n--- Step 4: Running Evaluation Suite ---")
    test_cases = convert_to_test_cases(all_questions)
    agent_outputs = convert_to_agent_outputs(results)

    eval_result = run_evaluation(test_cases, agent_outputs)

    # Final summary
    logger.info("\n--- Step 5: Final Summary ---")
    print("\n" + "=" * 60)
    print("COMPLETE EVALUATION MATRIX")
    print("=" * 60)

    print(f"\n{'Layer':<15} {'Score':>8} {'Status':<30}")
    print("-" * 55)

    layer_results = {}
    for layer_result in eval_result.layer_results:
        layer_name = layer_result.layer.value
        layer_score = layer_result.score
        layer_passed = layer_result.passed
        status = "PASS" if layer_passed else "FAIL"

        # Get failing metrics
        failing = [m.name for m in layer_result.metric_results if not m.passed]
        if failing:
            status += f" ({', '.join(failing)})"

        layer_results[layer_name] = {"score": layer_score, "passed": layer_passed, "status": status}
        print(f"{layer_name:<15} {layer_score:>7.1%} {status:<30}")

    print("-" * 55)
    print(f"{'OVERALL':<15} {eval_result.overall_score:>7.1%} {'PASS' if eval_result.overall_passed else 'FAIL':<30}")
    print("=" * 60)

    driver.close()
    return eval_result


if __name__ == "__main__":
    main()
