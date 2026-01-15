#!/usr/bin/env python3
"""
Test script for the Agent Evaluation Suite.

Tests:
1. Generates questions from database entities (answerable)
2. Creates rejection questions (not answerable)
3. Runs QA agent on all questions
4. Runs evaluation metrics
5. Generates reports
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add paths
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/benchmarks")

from neo4j import GraphDatabase
import os

# Neo4j config (same as agent_skb.py)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_sample_entities(driver, limit: int = 10) -> List[Dict[str, Any]]:
    """Get sample entities from Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND size(e.description) > 50
            RETURN e.name as name, e.ontology_type as type, e.description as description
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


def generate_in_db_questions(entities: List[Dict], relationships: List[Dict]) -> List[Dict[str, Any]]:
    """Generate questions that should be answerable from the database."""
    questions = []

    # Entity-based questions
    for i, entity in enumerate(entities[:5]):
        questions.append({
            "id": f"in_db_entity_{i+1}",
            "question": f"What is {entity['name']}?",
            "expected_entities": [entity['name']],
            "type": "retrieval",
            "difficulty": "easy",
            "should_reject": False,
            "metadata": {"source": "entity", "entity_type": entity['type']}
        })

    # Relationship-based questions
    for i, rel in enumerate(relationships[:5]):
        questions.append({
            "id": f"in_db_rel_{i+1}",
            "question": f"What is the relationship between {rel['source']} and {rel['target']}?",
            "expected_entities": [rel['source'], rel['target']],
            "expected_relationships": [{"source": rel['source'], "type": rel['rel_type'], "target": rel['target']}],
            "type": "retrieval",
            "difficulty": "medium",
            "should_reject": False,
            "metadata": {"source": "relationship", "rel_type": rel['rel_type']}
        })

    # Complex multi-hop questions
    questions.append({
        "id": "in_db_complex_1",
        "question": "How does Aggregate Demand influence the economy?",
        "expected_entities": ["Aggregate Demand"],
        "type": "agentic",
        "difficulty": "hard",
        "should_reject": False,
        "metadata": {"source": "complex"}
    })

    questions.append({
        "id": "in_db_complex_2",
        "question": "What factors cause inflation?",
        "expected_entities": ["Inflation"],
        "type": "agentic",
        "difficulty": "hard",
        "should_reject": False,
        "metadata": {"source": "complex"}
    })

    return questions


def generate_rejection_questions() -> List[Dict[str, Any]]:
    """Generate questions that should be rejected (not in database)."""
    return [
        {
            "id": "reject_realtime_1",
            "question": "What is the current stock price of Apple?",
            "type": "rejection",
            "difficulty": "easy",
            "should_reject": True,
            "rejection_reason": "Real-time stock data not available in knowledge graph",
            "metadata": {"category": "real_time_data"}
        },
        {
            "id": "reject_future_1",
            "question": "Who will win the 2028 presidential election?",
            "type": "rejection",
            "difficulty": "easy",
            "should_reject": True,
            "rejection_reason": "Future events cannot be predicted",
            "metadata": {"category": "future_prediction"}
        },
        {
            "id": "reject_personal_1",
            "question": "What is my home address?",
            "type": "rejection",
            "difficulty": "easy",
            "should_reject": True,
            "rejection_reason": "Personal information not stored",
            "metadata": {"category": "personal_info"}
        },
        {
            "id": "reject_fictional_1",
            "question": "What is the GDP of Wakanda?",
            "type": "rejection",
            "difficulty": "easy",
            "should_reject": True,
            "rejection_reason": "Fictional location",
            "metadata": {"category": "fictional"}
        },
        {
            "id": "reject_nonsense_1",
            "question": "What color is the number seven when it sneezes?",
            "type": "rejection",
            "difficulty": "easy",
            "should_reject": True,
            "rejection_reason": "Nonsensical question",
            "metadata": {"category": "nonsense"}
        }
    ]


def run_qa_agent(questions: List[Dict[str, Any]]) -> List[Tuple[Dict, Any]]:
    """Run QA agent on questions and capture outputs."""
    from agent_qa import ReActQAAgent

    logger.info("Initializing QA Agent...")
    agent = ReActQAAgent()

    results = []
    for i, q in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {q['question'][:50]}...")

        try:
            response = agent.answer_question(q['question'])
            results.append((q, response))

            logger.info(f"  Answer: {response.answer[:100]}...")
            logger.info(f"  Confidence: {response.confidence:.2f}")
            logger.info(f"  Citations: {len(response.citations)}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append((q, None))

    return results


def convert_to_test_cases(questions: List[Dict[str, Any]]):
    """Convert questions to TestCase format for evaluation."""
    from agent_eval.metrics.base import TestCase

    test_cases = []
    for q in questions:
        tc = TestCase(
            id=q['id'],
            question=q['question'],
            expected_entities=q.get('expected_entities', []),
            expected_relationships=q.get('expected_relationships', []),
            type=q.get('type', 'all'),
            difficulty=q.get('difficulty', 'medium'),
            should_reject=q.get('should_reject', False),
            rejection_reason=q.get('rejection_reason'),
            metadata=q.get('metadata', {}),
        )
        test_cases.append(tc)
    return test_cases


def _extract_step_data(step) -> Dict[str, Any]:
    """Extract thought step data from either dict or ThoughtStep object."""
    if isinstance(step, dict):
        # Already a dict (from LangGraph state)
        thought = step.get("thought", "")
        action = step.get("action")
        observation = step.get("observation", "")

        if action:
            if isinstance(action, dict):
                action_data = {
                    "tool_name": action.get("tool_name", ""),
                    "arguments": action.get("arguments", {}),
                }
            else:
                # ToolCall object
                action_data = {
                    "tool_name": getattr(action, 'tool_name', ''),
                    "arguments": getattr(action, 'arguments', {}),
                }
        else:
            action_data = None
    else:
        # ThoughtStep object
        thought = getattr(step, 'thought', '')
        observation = getattr(step, 'observation', '')
        action = getattr(step, 'action', None)

        if action:
            action_data = {
                "tool_name": getattr(action, 'tool_name', ''),
                "arguments": getattr(action, 'arguments', {}),
            }
        else:
            action_data = None

    return {
        "thought": thought,
        "action": action_data,
        "observation": observation,
    }


def _extract_context_items(response) -> List[Dict[str, Any]]:
    """Extract context items from reasoning steps to populate visited entities."""
    context_items = []

    # Extract entity lookups from reasoning steps and add as context items
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
                # Extract entities mentioned in Cypher query results
                query = action.get("arguments", {}).get("query", "")
                context_items.append({
                    "source_type": "neo4j",
                    "source_id": "cypher_query",
                    "content": observation,
                    "relevance_score": 1.0,
                    "metadata": {"query": query},
                })

    # Also add from citations if they're graph-based
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


def convert_to_agent_outputs(results: List[Tuple[Dict, Any]]):
    """Convert QA responses to AgentOutput format for evaluation."""
    from agent_eval.metrics.base import AgentOutput

    outputs = []
    for q, response in results:
        if response is None:
            output = AgentOutput(
                question=q['question'],
                answer="ERROR: Agent failed",
                confidence=0.0,
            )
        else:
            # Extract citations properly (handle both dict and object)
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

            # Extract thought history
            thought_history = []
            if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
                for step in response.reasoning_steps:
                    thought_history.append(_extract_step_data(step))

            # Extract context items for graph traversal metric
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
    from agent_eval import EvalConfig, AgentEvaluationRunner
    from agent_eval.runner import create_default_runner
    from agent_eval.reporting import HumanReporter, JSONReporter

    logger.info("Running evaluation suite...")

    config = EvalConfig()
    runner = create_default_runner(config)

    result = runner.run_evaluation(test_cases, agent_outputs)

    # Print human-readable report
    reporter = HumanReporter(use_colors=True)
    reporter.print_report(result)

    # Save JSON report
    json_reporter = JSONReporter("benchmarks/agent_eval/reports")
    report_path = json_reporter.generate(result, "test_run_report.json")
    logger.info(f"JSON report saved to {report_path}")

    return result


def main():
    """Main test execution."""
    logger.info("=" * 60)
    logger.info("AGENT EVALUATION SUITE TEST")
    logger.info("=" * 60)

    # Step 1: Connect to Neo4j and get sample data
    logger.info("\n--- Step 1: Querying Neo4j for sample data ---")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    entities = get_sample_entities(driver, limit=10)
    relationships = get_sample_relationships(driver, limit=10)

    logger.info(f"Found {len(entities)} sample entities")
    logger.info(f"Found {len(relationships)} sample relationships")

    # Step 2: Generate questions
    logger.info("\n--- Step 2: Generating test questions ---")
    in_db_questions = generate_in_db_questions(entities, relationships)
    rejection_questions = generate_rejection_questions()
    all_questions = in_db_questions + rejection_questions

    logger.info(f"Generated {len(in_db_questions)} in-database questions")
    logger.info(f"Generated {len(rejection_questions)} rejection questions")
    logger.info(f"Total: {len(all_questions)} questions")

    # Print questions
    print("\n=== IN-DATABASE QUESTIONS ===")
    for q in in_db_questions:
        print(f"  [{q['id']}] {q['question']}")

    print("\n=== REJECTION QUESTIONS ===")
    for q in rejection_questions:
        print(f"  [{q['id']}] {q['question']}")

    # Step 3: Run QA agent
    logger.info("\n--- Step 3: Running QA Agent ---")
    results = run_qa_agent(all_questions)

    # Print results summary
    print("\n=== QA AGENT RESULTS ===")
    for q, response in results:
        status = "OK" if response else "FAILED"
        if response:
            conf = f"{response.confidence:.2f}"
            answer_preview = response.answer[:80].replace('\n', ' ')
            external = "web" if response.external_info_used else "graph"
        else:
            conf = "N/A"
            answer_preview = "ERROR"
            external = "N/A"
        print(f"  [{status}] {q['id']}: conf={conf}, source={external}")
        print(f"       Q: {q['question'][:60]}...")
        print(f"       A: {answer_preview}...")

    # Step 4: Run evaluation
    logger.info("\n--- Step 4: Running Evaluation Suite ---")
    test_cases = convert_to_test_cases(all_questions)
    agent_outputs = convert_to_agent_outputs(results)

    eval_result = run_evaluation(test_cases, agent_outputs)

    # Step 5: Summary
    logger.info("\n--- Step 5: Summary ---")
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(all_questions)}")
    print(f"In-database: {len(in_db_questions)}")
    print(f"Rejection: {len(rejection_questions)}")
    print(f"Overall score: {eval_result.overall_score:.2%}")
    print(f"Overall passed: {eval_result.overall_passed}")

    driver.close()
    return eval_result


if __name__ == "__main__":
    main()
