"""
ReAct DPO data generator for faithful reasoning training.

Generates preference pairs with:
- Chosen: Complete Thought→Action→Observation chains, grounded answers
- Rejected: Premature conclusions, hallucinated reasoning, ignored tool results

Perturbation types specific to ReAct:
1. skip_verification - Answer without using tools
2. ignore_observation - Contradict tool output in reasoning
3. wrong_tool_selection - Use inappropriate tool for the task
4. premature_conclusion - Stop reasoning too early
5. hallucinated_reasoning - Non-sequitur or fabricated thoughts
"""

import os
import re
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActPerturbationType(str, Enum):
    """Types of perturbations for generating unfaithful ReAct traces."""
    SKIP_VERIFICATION = "skip_verification"
    IGNORE_OBSERVATION = "ignore_observation"
    WRONG_TOOL_SELECTION = "wrong_tool_selection"
    PREMATURE_CONCLUSION = "premature_conclusion"
    HALLUCINATED_REASONING = "hallucinated_reasoning"
    MISSING_THOUGHT = "missing_thought"
    WRONG_ENTITY_REFERENCE = "wrong_entity_reference"


@dataclass
class ReActStep:
    """A single step in a ReAct trace."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

    def to_text(self) -> str:
        """Convert step to text format."""
        parts = [f"Thought: {self.thought}"]
        if self.action:
            parts.append(f"Action: {self.action}")
            if self.action_input:
                parts.append(f"Action Input: {json.dumps(self.action_input)}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


@dataclass
class ReActTrace:
    """A complete ReAct trace with steps and final answer."""
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    citations: List[Dict[str, str]] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert trace to text format."""
        parts = [step.to_text() for step in self.steps]
        if self.final_answer:
            parts.append(f"Thought: I now have enough information to answer the question.")
            parts.append(f"Final Answer: {self.final_answer}")
        return "\n\n".join(parts)


@dataclass
class ReActDPOSample:
    """A single DPO training sample for ReAct."""
    id: str
    prompt: str  # Question + optional context
    chosen: str  # Faithful ReAct trace
    rejected: str  # Unfaithful ReAct trace
    perturbation_type: ReActPerturbationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "perturbation_type": self.perturbation_type.value,
            "metadata": self.metadata,
        }


class ReActPerturbationGenerator:
    """Generate unfaithful ReAct traces through perturbations."""

    # Available tools for the agent
    AVAILABLE_TOOLS = [
        "graph_lookup",
        "entity_resolve",
        "cypher_query",
        "web_search",
        "get_entity_relationships",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def skip_verification(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that skips tool verification.

        Creates a trace that immediately answers without using any tools.
        """
        # Extract any entity mentions from the question
        hallucinated_answer = self._generate_confident_but_wrong_answer(
            question, trace.final_answer
        )

        rejected_trace = f"""Thought: I can answer this question directly from my knowledge.

Final Answer: {hallucinated_answer}"""

        return rejected_trace, ReActPerturbationType.SKIP_VERIFICATION

    def ignore_observation(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that ignores or contradicts tool observations.

        Uses tools but draws conclusions that contradict what was found.
        """
        if not trace.steps:
            return self.skip_verification(trace, question)

        # Copy steps but modify the final reasoning
        rejected_parts = []

        for i, step in enumerate(trace.steps):
            if i == len(trace.steps) - 1 and step.observation:
                # Keep the observation but add contradicting reasoning
                rejected_parts.append(step.to_text())
                rejected_parts.append(
                    f"\nThought: The tool results don't seem accurate. "
                    f"Based on my knowledge, this information is incorrect. "
                    f"Let me provide the correct answer instead."
                )
            else:
                rejected_parts.append(step.to_text())

        # Generate answer that contradicts observation
        contradicted_answer = self._generate_contradicting_answer(
            trace.steps[-1].observation if trace.steps else "",
            trace.final_answer
        )

        rejected_parts.append(f"\nFinal Answer: {contradicted_answer}")

        return "\n\n".join(rejected_parts), ReActPerturbationType.IGNORE_OBSERVATION

    def wrong_tool_selection(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that uses inappropriate tools.

        Selects tools that don't match the task requirements.
        """
        if not trace.steps:
            return self.skip_verification(trace, question)

        rejected_parts = []

        for step in trace.steps:
            if step.action:
                # Replace with wrong tool
                wrong_tool = self._get_wrong_tool(step.action)
                rejected_parts.append(
                    f"Thought: {step.thought}\n"
                    f"Action: {wrong_tool}\n"
                    f"Action Input: {json.dumps(step.action_input or {})}"
                )
                if step.observation:
                    # Generate confused observation
                    rejected_parts.append(
                        f"Observation: The tool returned unexpected results. "
                        f"This doesn't seem right but I'll continue anyway."
                    )
            else:
                rejected_parts.append(step.to_text())

        # Add confused final answer
        rejected_parts.append(
            f"\nThought: The results are confusing but I'll try to answer.\n"
            f"Final Answer: {trace.final_answer} (Note: I'm not entirely sure about this.)"
        )

        return "\n\n".join(rejected_parts), ReActPerturbationType.WRONG_TOOL_SELECTION

    def premature_conclusion(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that stops reasoning too early.

        Only uses first step/observation and jumps to conclusion.
        """
        if len(trace.steps) <= 1:
            return self.skip_verification(trace, question)

        # Only take the first step
        first_step = trace.steps[0]
        rejected_parts = [first_step.to_text()]

        # Jump to premature conclusion
        incomplete_answer = self._generate_incomplete_answer(
            first_step.observation or "",
            trace.final_answer
        )

        rejected_parts.append(
            f"\nThought: I have enough information now.\n"
            f"Final Answer: {incomplete_answer}"
        )

        return "\n\n".join(rejected_parts), ReActPerturbationType.PREMATURE_CONCLUSION

    def hallucinated_reasoning(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response with non-sequitur or fabricated thoughts.

        Thoughts don't logically follow from observations.
        """
        if not trace.steps:
            return self.skip_verification(trace, question)

        rejected_parts = []

        for step in trace.steps:
            # Keep action and observation but replace thought with hallucinated one
            hallucinated_thought = self._generate_hallucinated_thought(
                step.thought, step.observation
            )

            new_step = ReActStep(
                thought=hallucinated_thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
            )
            rejected_parts.append(new_step.to_text())

        # Generate answer with fabricated reasoning
        rejected_parts.append(
            f"\nThought: Based on the historical precedent and common patterns, "
            f"I can confidently conclude the answer.\n"
            f"Final Answer: {trace.final_answer} This is well-documented in multiple sources."
        )

        return "\n\n".join(rejected_parts), ReActPerturbationType.HALLUCINATED_REASONING

    def missing_thought(
        self,
        trace: ReActTrace,
        question: str,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that skips thought steps.

        Jumps directly to actions without reasoning.
        """
        if not trace.steps:
            return self.skip_verification(trace, question)

        rejected_parts = []

        for step in trace.steps:
            if step.action:
                # Skip thought, go directly to action
                rejected_parts.append(
                    f"Action: {step.action}\n"
                    f"Action Input: {json.dumps(step.action_input or {})}"
                )
                if step.observation:
                    rejected_parts.append(f"Observation: {step.observation}")
            # Skip thought-only steps

        rejected_parts.append(f"\nFinal Answer: {trace.final_answer}")

        return "\n\n".join(rejected_parts), ReActPerturbationType.MISSING_THOUGHT

    def wrong_entity_reference(
        self,
        trace: ReActTrace,
        question: str,
        entities: List[Dict[str, Any]] = None,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Generate response that references wrong entities.

        Confuses entities or attributes information to wrong sources.
        """
        if not entities or len(entities) < 2:
            return self.hallucinated_reasoning(trace, question)

        # Pick two entities to swap
        e1, e2 = self.rng.sample(entities[:min(5, len(entities))], 2)
        name1 = e1.get("name", "Entity1")
        name2 = e2.get("name", "Entity2")

        # Swap entity names in the trace
        trace_text = trace.to_text()
        placeholder = "___SWAP_PLACEHOLDER___"
        swapped = trace_text.replace(name1, placeholder)
        swapped = swapped.replace(name2, name1)
        swapped = swapped.replace(placeholder, name2)

        return swapped, ReActPerturbationType.WRONG_ENTITY_REFERENCE

    def _get_wrong_tool(self, correct_tool: str) -> str:
        """Get an inappropriate tool for the task."""
        other_tools = [t for t in self.AVAILABLE_TOOLS if t != correct_tool]
        if other_tools:
            return self.rng.choice(other_tools)
        return "unknown_tool"

    def _generate_confident_but_wrong_answer(
        self,
        question: str,
        correct_answer: str,
    ) -> str:
        """Generate a confident but incorrect answer."""
        prefixes = [
            "Based on my knowledge, ",
            "From what I understand, ",
            "It's well established that ",
            "The answer is clearly that ",
        ]

        # Modify the correct answer slightly
        modifications = [
            lambda a: a.replace(" is ", " is not ") if " is " in a else f"not {a}",
            lambda a: a.replace("first", "last").replace("last", "first"),
            lambda a: a + " However, this is disputed by some sources.",
            lambda a: "There is no definitive answer, but " + a,
        ]

        prefix = self.rng.choice(prefixes)
        modifier = self.rng.choice(modifications)

        return prefix + modifier(correct_answer)

    def _generate_contradicting_answer(
        self,
        observation: str,
        correct_answer: str,
    ) -> str:
        """Generate an answer that contradicts the observation."""
        contradictions = [
            f"Despite what the data shows, {correct_answer.replace('is', 'is not')}",
            f"The tool results are incorrect. The actual answer is different from {correct_answer}.",
            f"This contradicts the standard understanding. {correct_answer}",
        ]
        return self.rng.choice(contradictions)

    def _generate_incomplete_answer(
        self,
        partial_observation: str,
        correct_answer: str,
    ) -> str:
        """Generate an incomplete answer based on partial information."""
        # Take first portion of correct answer
        words = correct_answer.split()
        if len(words) > 10:
            incomplete = " ".join(words[:len(words)//2]) + "..."
        else:
            incomplete = correct_answer

        return f"Based on initial findings, {incomplete}"

    def _generate_hallucinated_thought(
        self,
        original_thought: str,
        observation: str,
    ) -> str:
        """Generate a thought that doesn't follow from the observation."""
        hallucinated_patterns = [
            "This reminds me of a similar case involving historical precedents. ",
            "According to general principles of the field, ",
            "My intuition suggests that despite what the data shows, ",
            "There's a well-known pattern that indicates ",
            "From my training data, I recall that ",
        ]

        pattern = self.rng.choice(hallucinated_patterns)
        # Keep part of original thought to maintain some coherence
        original_words = original_thought.split()
        if len(original_words) > 5:
            kept = " ".join(original_words[:5])
            return pattern + kept + "..."
        return pattern + original_thought

    def perturb(
        self,
        trace: ReActTrace,
        question: str,
        perturbation_type: Optional[ReActPerturbationType] = None,
        entities: List[Dict[str, Any]] = None,
    ) -> Tuple[str, ReActPerturbationType]:
        """
        Apply a perturbation to create an unfaithful ReAct trace.

        Args:
            trace: Original faithful ReAct trace
            question: The question being answered
            perturbation_type: Specific perturbation to apply, or None for random
            entities: Optional list of entities for entity-based perturbations

        Returns:
            Tuple of (perturbed_trace_text, perturbation_type_used)
        """
        if perturbation_type is None:
            perturbation_type = self.rng.choice(list(ReActPerturbationType))

        if perturbation_type == ReActPerturbationType.SKIP_VERIFICATION:
            return self.skip_verification(trace, question)
        elif perturbation_type == ReActPerturbationType.IGNORE_OBSERVATION:
            return self.ignore_observation(trace, question)
        elif perturbation_type == ReActPerturbationType.WRONG_TOOL_SELECTION:
            return self.wrong_tool_selection(trace, question)
        elif perturbation_type == ReActPerturbationType.PREMATURE_CONCLUSION:
            return self.premature_conclusion(trace, question)
        elif perturbation_type == ReActPerturbationType.HALLUCINATED_REASONING:
            return self.hallucinated_reasoning(trace, question)
        elif perturbation_type == ReActPerturbationType.MISSING_THOUGHT:
            return self.missing_thought(trace, question)
        elif perturbation_type == ReActPerturbationType.WRONG_ENTITY_REFERENCE:
            return self.wrong_entity_reference(trace, question, entities)
        else:
            return self.skip_verification(trace, question)


class ReActDPODataGenerator:
    """
    Generate DPO training data for ReAct faithfulness.

    Two approaches:
    1. LLM-generated unfaithful traces
    2. Perturbation-based modifications of correct traces
    """

    def __init__(
        self,
        teacher_llm=None,
        seed: int = 42,
    ):
        self.teacher_llm = teacher_llm
        self.rng = random.Random(seed)
        self.perturbation_gen = ReActPerturbationGenerator(seed)
        self._sample_counter = 0

    def _generate_sample_id(self, method: str) -> str:
        """Generate a unique sample ID."""
        self._sample_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"react_dpo_{method}_{timestamp}_{self._sample_counter}"

    def parse_react_trace(self, trace_text: str) -> ReActTrace:
        """
        Parse a ReAct trace from text format.

        Args:
            trace_text: Raw text of the ReAct trace

        Returns:
            Parsed ReActTrace object
        """
        trace = ReActTrace()
        current_step = None
        final_answer = ""

        lines = trace_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Thought:"):
                # Save previous step if exists
                if current_step is not None:
                    trace.steps.append(current_step)
                thought = line.replace("Thought:", "").strip()
                current_step = ReActStep(thought=thought)

            elif line.startswith("Action:"):
                if current_step:
                    current_step.action = line.replace("Action:", "").strip()

            elif line.startswith("Action Input:"):
                if current_step:
                    try:
                        input_str = line.replace("Action Input:", "").strip()
                        current_step.action_input = json.loads(input_str)
                    except json.JSONDecodeError:
                        current_step.action_input = {"raw": input_str}

            elif line.startswith("Observation:"):
                if current_step:
                    current_step.observation = line.replace("Observation:", "").strip()

            elif line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()

        # Add last step
        if current_step is not None:
            trace.steps.append(current_step)

        trace.final_answer = final_answer
        return trace

    def generate_from_perturbation(
        self,
        question: str,
        faithful_trace: ReActTrace,
        context: str = "",
        entities: List[Dict[str, Any]] = None,
        perturbation_type: Optional[ReActPerturbationType] = None,
    ) -> ReActDPOSample:
        """
        Generate DPO sample using perturbation.

        Args:
            question: User question
            faithful_trace: Correct, grounded ReAct trace
            context: Optional context
            entities: Entities mentioned in the trace
            perturbation_type: Specific perturbation to use

        Returns:
            ReActDPOSample with chosen and rejected traces
        """
        # Generate rejected trace through perturbation
        rejected, pert_type = self.perturbation_gen.perturb(
            trace=faithful_trace,
            question=question,
            perturbation_type=perturbation_type,
            entities=entities,
        )

        # Build prompt
        prompt_parts = [f"Question: {question}"]
        if context:
            prompt_parts.insert(0, f"Context:\n{context[:2000]}")
        prompt = "\n\n".join(prompt_parts)

        return ReActDPOSample(
            id=self._generate_sample_id("perturbation"),
            prompt=prompt,
            chosen=faithful_trace.to_text(),
            rejected=rejected,
            perturbation_type=pert_type,
            metadata={
                "method": "perturbation",
                "num_steps": len(faithful_trace.steps),
                "has_context": bool(context),
            },
        )

    def generate_from_llm(
        self,
        question: str,
        faithful_trace: ReActTrace,
        context: str = "",
        perturbation_type: Optional[ReActPerturbationType] = None,
    ) -> Optional[ReActDPOSample]:
        """
        Generate DPO sample using LLM-generated unfaithful trace.

        Args:
            question: User question
            faithful_trace: Correct, grounded ReAct trace
            context: Optional context
            perturbation_type: Type of unfaithfulness to generate

        Returns:
            ReActDPOSample or None if LLM not available
        """
        if self.teacher_llm is None:
            logger.warning("Teacher LLM not configured")
            return None

        pert_type = perturbation_type or self.rng.choice(list(ReActPerturbationType))

        # Create prompt for generating unfaithful trace
        generation_prompt = f"""You are generating training data for detecting unfaithful AI reasoning.

Given a question and a correct ReAct reasoning trace, generate an UNFAITHFUL version that demonstrates: {pert_type.value}

Question: {question}

Context (if any):
{context[:1500] if context else "None"}

Correct ReAct Trace:
{faithful_trace.to_text()}

Generate an UNFAITHFUL ReAct trace that demonstrates {pert_type.value}:
- {self._get_perturbation_instructions(pert_type)}

Generate the unfaithful trace now (format: Thought/Action/Observation/Final Answer):"""

        try:
            response = self.teacher_llm.invoke(generation_prompt)
            if hasattr(response, 'content'):
                rejected = response.content
            else:
                rejected = str(response)

            # Build prompt
            prompt_parts = [f"Question: {question}"]
            if context:
                prompt_parts.insert(0, f"Context:\n{context[:2000]}")
            prompt = "\n\n".join(prompt_parts)

            return ReActDPOSample(
                id=self._generate_sample_id("llm"),
                prompt=prompt,
                chosen=faithful_trace.to_text(),
                rejected=rejected,
                perturbation_type=pert_type,
                metadata={
                    "method": "llm_generated",
                    "num_steps": len(faithful_trace.steps),
                },
            )

        except Exception as e:
            logger.warning(f"LLM trace generation failed: {e}")
            return None

    def _get_perturbation_instructions(self, pert_type: ReActPerturbationType) -> str:
        """Get instructions for generating specific perturbation type."""
        instructions = {
            ReActPerturbationType.SKIP_VERIFICATION:
                "Answer immediately without using any tools. Be confident but potentially wrong.",
            ReActPerturbationType.IGNORE_OBSERVATION:
                "Use tools but then ignore or contradict what they return in your reasoning.",
            ReActPerturbationType.WRONG_TOOL_SELECTION:
                "Use inappropriate tools for the task (e.g., web_search when graph_lookup is needed).",
            ReActPerturbationType.PREMATURE_CONCLUSION:
                "Stop after one tool call and conclude prematurely without gathering enough information.",
            ReActPerturbationType.HALLUCINATED_REASONING:
                "Include thoughts that don't logically follow from observations, citing non-existent patterns.",
            ReActPerturbationType.MISSING_THOUGHT:
                "Skip the Thought steps and jump directly to actions without explaining reasoning.",
            ReActPerturbationType.WRONG_ENTITY_REFERENCE:
                "Confuse entities or attribute information to the wrong sources.",
        }
        return instructions.get(pert_type, "Demonstrate unfaithful reasoning.")

    def generate_from_conversation(
        self,
        conversation: Dict[str, Any],
        use_llm: bool = False,
    ) -> Optional[ReActDPOSample]:
        """
        Generate DPO sample from a recorded agent conversation.

        Args:
            conversation: Dict with question, trace, context, etc.
            use_llm: Whether to use LLM for rejected trace

        Returns:
            ReActDPOSample or None if generation fails
        """
        question = conversation.get("question", "")
        trace_text = conversation.get("trace", "")
        context = conversation.get("context", "")
        entities = conversation.get("entities", [])

        if not question or not trace_text:
            return None

        # Parse the trace
        trace = self.parse_react_trace(trace_text)

        if not trace.steps and not trace.final_answer:
            logger.warning("Failed to parse trace")
            return None

        # Generate sample
        if use_llm and self.teacher_llm:
            sample = self.generate_from_llm(
                question=question,
                faithful_trace=trace,
                context=context,
            )
            if sample:
                return sample

        # Fallback to perturbation
        return self.generate_from_perturbation(
            question=question,
            faithful_trace=trace,
            context=context,
            entities=entities,
        )

    def generate_dataset(
        self,
        conversations: List[Dict[str, Any]],
        use_llm: bool = False,
        llm_ratio: float = 0.3,
        perturbation_distribution: Optional[Dict[ReActPerturbationType, float]] = None,
    ) -> List[ReActDPOSample]:
        """
        Generate DPO dataset from recorded conversations.

        Args:
            conversations: List of conversation dicts
            use_llm: Whether to use LLM for some rejected traces
            llm_ratio: Ratio of samples to generate with LLM
            perturbation_distribution: Optional distribution over perturbation types

        Returns:
            List of ReActDPOSample objects
        """
        samples = []

        for conv in conversations:
            # Decide method
            use_llm_for_sample = use_llm and self.rng.random() < llm_ratio

            sample = self.generate_from_conversation(conv, use_llm=use_llm_for_sample)
            if sample:
                samples.append(sample)

        logger.info(f"Generated {len(samples)}/{len(conversations)} ReAct DPO samples")

        # Log perturbation type distribution
        type_counts = {}
        for s in samples:
            pt = s.perturbation_type.value
            type_counts[pt] = type_counts.get(pt, 0) + 1
        logger.info(f"Perturbation distribution: {type_counts}")

        return samples

    def save_dataset(
        self,
        samples: List[ReActDPOSample],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """
        Save generated samples to file.

        Args:
            samples: List of ReActDPOSample objects
            output_path: Path to output file
            format: Output format (jsonl or json)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump([s.to_dict() for s in samples], f, indent=2)

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    def load_conversations(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load conversations from file.

        Args:
            input_path: Path to JSONL file with conversations

        Returns:
            List of conversation dicts
        """
        conversations = []
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        return conversations


def generate_react_dpo_data(
    input_path: str,
    output_path: str,
    teacher_llm=None,
    use_llm: bool = False,
    llm_ratio: float = 0.3,
    seed: int = 42,
) -> List[ReActDPOSample]:
    """
    Convenience function to generate ReAct DPO training data.

    Args:
        input_path: Path to input conversations (JSONL)
        output_path: Path to save generated data
        teacher_llm: Optional LLM for generating unfaithful traces
        use_llm: Whether to use LLM
        llm_ratio: Ratio of LLM-generated samples
        seed: Random seed

    Returns:
        List of generated samples
    """
    generator = ReActDPODataGenerator(teacher_llm=teacher_llm, seed=seed)

    # Load conversations
    conversations = generator.load_conversations(input_path)
    logger.info(f"Loaded {len(conversations)} conversations")

    # Generate samples
    samples = generator.generate_dataset(
        conversations=conversations,
        use_llm=use_llm,
        llm_ratio=llm_ratio,
    )

    # Save
    generator.save_dataset(samples, output_path)

    return samples
