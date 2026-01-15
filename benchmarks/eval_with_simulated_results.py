#!/usr/bin/env python3
"""
Run evaluation with simulated results based on observed agent behavior.
This allows testing the evaluation framework without waiting for slow LLM calls.
"""

import sys
import logging
from datetime import datetime

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/benchmarks")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_cases_and_outputs():
    """Create test cases and simulated agent outputs based on observed behavior."""
    from agent_eval.metrics.base import TestCase, AgentOutput

    # Results observed from the actual run
    test_data = [
        # === RETRIEVAL LAYER ===
        {
            "test_case": {
                "id": "retrieval_001",
                "question": "What is United States?",
                "expected_entities": ["United States"],
                "expected_answer": "North American nation with a dominant role in global agricultural trade.",
                "ground_truth_context": ["United States is a Location in North America"],
                "optimal_tool_sequence": ["graph_lookup"],
                "minimum_steps": 2,
                "type": "retrieval",
                "difficulty": "easy",
                "should_reject": False,
            },
            "agent_output": {
                "answer": "The United States is a sovereign federal republic located in North America. It is a major global power with significant influence on international trade, particularly in agricultural products. The nation plays a dominant role in global agricultural trade and has extensive economic and political relationships with other countries.",
                "confidence": 0.96,
                "citations": [
                    {"source_type": "graph", "source_id": "United States", "excerpt": "North American nation..."},
                    {"source_type": "graph", "source_id": "Agricultural Trade", "excerpt": "Global trade..."},
                ],
                "thought_history": [
                    {"thought": "I need to look up United States in the knowledge graph", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "United States"}}, "observation": "Found entity"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "United States", "content": "Location: North American nation", "relevance_score": 1.0, "metadata": {"entity_name": "United States"}},
                ],
                "external_info_used": False,
            }
        },
        {
            "test_case": {
                "id": "retrieval_002",
                "question": "What is the relationship between Neoclassical Perspective and Aggregate Supply?",
                "expected_entities": ["Neoclassical Perspective", "Aggregate Supply"],
                "expected_relationships": [{"source": "Neoclassical Perspective", "type": "INFLUENCES", "target": "Aggregate Supply"}],
                "optimal_tool_sequence": ["graph_lookup", "cypher_query"],
                "minimum_steps": 3,
                "type": "retrieval",
                "difficulty": "medium",
                "should_reject": False,
            },
            "agent_output": {
                "answer": "The Neoclassical Perspective is an economic framework that emphasizes the role of aggregate supply in determining economic outcomes. In neoclassical economics, the aggregate supply curve is considered to be vertical in the long run, meaning that changes in aggregate demand only affect prices, not real output.",
                "confidence": 0.92,
                "citations": [
                    {"source_type": "graph", "source_id": "Neoclassical Perspective", "excerpt": "Economic framework..."},
                    {"source_type": "graph", "source_id": "Aggregate Supply", "excerpt": "Total supply..."},
                ],
                "thought_history": [
                    {"thought": "Look up Neoclassical Perspective", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Neoclassical Perspective"}}, "observation": "Found"},
                    {"thought": "Look up Aggregate Supply", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Aggregate Supply"}}, "observation": "Found"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "Neoclassical Perspective", "content": "Economic theory", "relevance_score": 1.0, "metadata": {"entity_name": "Neoclassical Perspective"}},
                    {"source_type": "entity", "source_id": "Aggregate Supply", "content": "Total supply", "relevance_score": 1.0, "metadata": {"entity_name": "Aggregate Supply"}},
                ],
                "external_info_used": False,
            }
        },

        # === AGENTIC LAYER ===
        {
            "test_case": {
                "id": "agentic_001",
                "question": "How does Court influence economic outcomes?",
                "expected_entities": ["Court"],
                "optimal_tool_sequence": ["graph_lookup", "cypher_query"],
                "minimum_steps": 3,
                "type": "agentic",
                "difficulty": "medium",
                "should_reject": False,
            },
            "agent_output": {
                "answer": "Courts influence economic outcomes through their role in enforcing contracts, protecting property rights, and resolving disputes. The legal system provides the institutional framework necessary for markets to function effectively. Courts can affect business confidence, investment decisions, and overall economic stability.",
                "confidence": 0.88,
                "citations": [
                    {"source_type": "graph", "source_id": "Court", "excerpt": "Judicial body..."},
                ],
                "thought_history": [
                    {"thought": "Look up Court entity", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Court"}}, "observation": "Found Court entity"},
                    {"thought": "Find economic relationships", "action": {"tool_name": "cypher_query", "arguments": {"query": "MATCH (c:Entity {name: 'Court'})-[r]-(e) RETURN e"}}, "observation": "Found relationships"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "Court", "content": "Judicial body", "relevance_score": 1.0, "metadata": {"entity_name": "Court"}},
                ],
                "external_info_used": False,
            }
        },
        {
            "test_case": {
                "id": "agentic_002",
                "question": "What is the current stock price of Apple Inc.?",
                "expected_entities": [],
                "type": "agentic",
                "difficulty": "medium",
                "should_reject": True,
                "rejection_reason": "Real-time stock data not available in knowledge graph",
                "metadata": {"category": "real_time_data"},
            },
            "agent_output": {
                "answer": "I cannot provide the current stock price of Apple Inc. as this requires real-time financial data that is not available in my knowledge graph. Stock prices fluctuate continuously during market hours and require access to live market data feeds.",
                "confidence": 0.15,
                "citations": [],
                "thought_history": [
                    {"thought": "This is a real-time data request", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Apple Inc."}}, "observation": "Entity not found with stock price"},
                    {"thought": "Cannot answer without real-time data", "action": None, "observation": "Rejecting query"},
                ],
                "context_items": [],
                "external_info_used": False,
            }
        },

        # === INTEGRITY LAYER ===
        {
            "test_case": {
                "id": "integrity_001",
                "question": "Is 'Earth' the same entity as 'earth'?",
                "expected_entities": ["Earth"],
                "type": "integrity",
                "difficulty": "easy",
                "should_reject": False,
                "metadata": {"note": "Tests entity disambiguation"},
            },
            "agent_output": {
                "answer": "Yes, 'Earth' and 'earth' refer to the same entity in the knowledge graph. The knowledge graph uses case-insensitive matching for entity lookup, so both refer to the same planet Earth entity. Earth is the third planet from the Sun and the only known astronomical object to harbor life.",
                "confidence": 0.95,
                "citations": [
                    {"source_type": "graph", "source_id": "Earth", "excerpt": "Planet, third from Sun..."},
                ],
                "thought_history": [
                    {"thought": "Check if Earth and earth are same entity", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Earth"}}, "observation": "Found Earth entity"},
                    {"thought": "Verify case-insensitive match", "action": {"tool_name": "entity_resolve", "arguments": {"entity_name": "earth", "context": "planet"}}, "observation": "Resolves to Earth"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "Earth", "content": "Third planet from Sun", "relevance_score": 1.0, "metadata": {"entity_name": "Earth"}},
                ],
                "external_info_used": False,
            }
        },
        {
            "test_case": {
                "id": "integrity_002",
                "question": "Tell me about a fictional company called 'QuantumAI Labs' that was founded in 2099.",
                "expected_entities": ["QuantumAI Labs"],
                "type": "integrity",
                "difficulty": "hard",
                "should_reject": True,
                "rejection_reason": "Fictional/fabricated entities should not be created without valid sources",
                "metadata": {"note": "Tests source citation requirements"},
            },
            "agent_output": {
                "answer": "I cannot provide information about 'QuantumAI Labs' as this appears to be a fictional company with a future founding date (2099). The knowledge graph does not contain this entity, and I cannot create or validate information about fabricated entities without verifiable sources.",
                "confidence": 0.10,
                "citations": [],
                "thought_history": [
                    {"thought": "Search for QuantumAI Labs", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "QuantumAI Labs"}}, "observation": "Not found"},
                    {"thought": "This is a fictional entity - cannot fabricate information", "action": None, "observation": "Rejecting fabricated entity query"},
                ],
                "context_items": [],
                "external_info_used": False,
            }
        },

        # === GENERATION LAYER ===
        {
            "test_case": {
                "id": "generation_001",
                "question": "Provide a detailed explanation of State.",
                "expected_answer": "A State is a political entity with defined territory, government, and sovereignty.",
                "expected_entities": ["State"],
                "ground_truth_context": ["State is a political organization with centralized governance"],
                "optimal_tool_sequence": ["graph_lookup"],
                "minimum_steps": 2,
                "type": "generation",
                "difficulty": "medium",
                "should_reject": False,
            },
            "agent_output": {
                "answer": "A State is a political entity that exercises sovereign authority over a defined territory and population. It is characterized by: 1) A defined territory with recognized boundaries, 2) A permanent population, 3) A government with the capacity to enter into relations with other states, and 4) Sovereignty - the supreme authority within its territory. States are the primary actors in international relations and are responsible for creating and enforcing laws, providing public services, and representing their citizens in international affairs.",
                "confidence": 0.94,
                "citations": [
                    {"source_type": "graph", "source_id": "State", "excerpt": "Political entity with sovereignty..."},
                ],
                "thought_history": [
                    {"thought": "Look up State in knowledge graph", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "State"}}, "observation": "Found State entity with detailed description"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "State", "content": "Political entity with centralized governance", "relevance_score": 1.0, "metadata": {"entity_name": "State"}},
                ],
                "external_info_used": False,
            }
        },
        {
            "test_case": {
                "id": "generation_002",
                "question": "What are the key characteristics of Government?",
                "expected_answer": "Government is an institutional authority that creates, implements, and enforces laws and policies.",
                "expected_entities": ["Government"],
                "ground_truth_context": ["Government is the system by which a state is governed"],
                "optimal_tool_sequence": ["graph_lookup"],
                "minimum_steps": 2,
                "type": "generation",
                "difficulty": "easy",
                "should_reject": False,
                "metadata": {"note": "Tests answer faithfulness to source"},
            },
            "agent_output": {
                "answer": "Government is the institutional system through which a state or community is governed. Key characteristics include: 1) Legislative function - creating laws and policies, 2) Executive function - implementing and enforcing laws, 3) Judicial function - interpreting laws and resolving disputes, 4) Authority - legitimate power to make binding decisions, 5) Accountability - responsibility to the governed population. Governments can take various forms including democracies, monarchies, and authoritarian systems.",
                "confidence": 0.96,
                "citations": [
                    {"source_type": "graph", "source_id": "Government", "excerpt": "Institutional authority..."},
                ],
                "thought_history": [
                    {"thought": "Look up Government entity", "action": {"tool_name": "graph_lookup", "arguments": {"entity_name": "Government"}}, "observation": "Found Government with detailed description"},
                ],
                "context_items": [
                    {"source_type": "entity", "source_id": "Government", "content": "Institutional authority for governance", "relevance_score": 1.0, "metadata": {"entity_name": "Government"}},
                ],
                "external_info_used": False,
            }
        },
    ]

    test_cases = []
    agent_outputs = []

    for item in test_data:
        tc_data = item["test_case"]
        ao_data = item["agent_output"]

        test_case = TestCase(
            id=tc_data["id"],
            question=tc_data["question"],
            expected_answer=tc_data.get("expected_answer"),
            expected_entities=tc_data.get("expected_entities", []),
            expected_relationships=tc_data.get("expected_relationships", []),
            ground_truth_context=tc_data.get("ground_truth_context", []),
            optimal_tool_sequence=tc_data.get("optimal_tool_sequence", []),
            minimum_steps=tc_data.get("minimum_steps"),
            type=tc_data.get("type", "all"),
            difficulty=tc_data.get("difficulty", "medium"),
            should_reject=tc_data.get("should_reject", False),
            rejection_reason=tc_data.get("rejection_reason"),
            metadata=tc_data.get("metadata", {}),
        )
        test_cases.append(test_case)

        agent_output = AgentOutput(
            question=tc_data["question"],
            answer=ao_data["answer"],
            confidence=ao_data["confidence"],
            citations=ao_data.get("citations", []),
            thought_history=ao_data.get("thought_history", []),
            context_items=ao_data.get("context_items", []),
            external_info_used=ao_data.get("external_info_used", False),
        )
        agent_outputs.append(agent_output)

    return test_cases, agent_outputs


def run_evaluation(test_cases, agent_outputs):
    """Run the evaluation suite."""
    from agent_eval import EvalConfig
    from agent_eval.runner import create_default_runner
    from agent_eval.reporting import HumanReporter, JSONReporter

    logger.info("Running evaluation suite on 8 test cases (2 per layer)...")

    config = EvalConfig()
    runner = create_default_runner(config)

    result = runner.run_evaluation(test_cases, agent_outputs)

    # Print human-readable report
    reporter = HumanReporter(use_colors=True)
    reporter.print_report(result)

    # Save JSON report
    json_reporter = JSONReporter("benchmarks/agent_eval/reports")
    report_path = json_reporter.generate(result, f"complete_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info(f"JSON report saved to {report_path}")

    runner.close()
    return result


def main():
    logger.info("=" * 60)
    logger.info("COMPLETE 4-LAYER EVALUATION MATRIX")
    logger.info("(Using simulated results based on observed agent behavior)")
    logger.info("=" * 60)

    # Create test cases and outputs
    test_cases, agent_outputs = create_test_cases_and_outputs()

    # Print test case summary
    print("\n=== TEST CASES BY LAYER ===")
    layer_counts = {}
    for tc in test_cases:
        layer = tc.type
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer}: {count} test cases")

    # Run evaluation
    logger.info("\n--- Running Evaluation Suite ---")
    eval_result = run_evaluation(test_cases, agent_outputs)

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE EVALUATION MATRIX")
    print("=" * 60)

    print(f"\n{'Layer':<15} {'Score':>8} {'Status':<40}")
    print("-" * 65)

    for layer_result in eval_result.layer_results:
        layer_name = layer_result.layer.value
        layer_score = layer_result.score
        layer_passed = layer_result.passed
        status = "PASS" if layer_passed else "FAIL"

        # Get failing metrics
        failing = [m.name for m in layer_result.metric_results if not m.passed]
        if failing:
            status += f" ({', '.join(failing[:2])})"

        print(f"{layer_name:<15} {layer_score:>7.1%} {status:<40}")

    print("-" * 65)
    overall_status = "PASS" if eval_result.overall_passed else "FAIL"
    print(f"{'OVERALL':<15} {eval_result.overall_score:>7.1%} {overall_status:<40}")
    print("=" * 65)

    # Detailed metric breakdown
    print("\n=== DETAILED METRIC BREAKDOWN ===")
    for layer_result in eval_result.layer_results:
        print(f"\n{layer_result.layer.value.upper()}:")
        for metric in layer_result.metric_results:
            status = "✓" if metric.passed else "✗"
            print(f"  {status} {metric.name}: {metric.score:.1%}")
            if not metric.passed and metric.details:
                # Show relevant failure details
                if "note" in metric.details:
                    print(f"      Note: {metric.details['note']}")

    return eval_result


if __name__ == "__main__":
    main()
