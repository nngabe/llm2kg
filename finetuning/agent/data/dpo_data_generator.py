"""
DPO data generator for faithfulness training.

Generates preference pairs with:
- Chosen: Faithful responses grounded in retrieved context
- Rejected: Hallucinated or unfaithful responses

Uses two approaches:
1. LLM-generated hallucinations
2. Perturbation-based modifications of correct answers
"""

import os
import re
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Perturbation types for generating unfaithful responses
PerturbationType = Literal[
    "swap_entity_names",
    "invent_relationship",
    "wrong_attribute",
    "mix_entities",
    "add_unsupported_claim",
    "contradict_context",
]


@dataclass
class DPOSample:
    """A single DPO training sample."""
    id: str
    prompt: str  # Context + Question
    chosen: str  # Faithful response
    rejected: str  # Hallucinated/unfaithful response
    perturbation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "perturbation_type": self.perturbation_type,
            "metadata": self.metadata,
        }


class PerturbationGenerator:
    """Generate unfaithful responses through perturbations."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def swap_entity_names(
        self,
        response: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Swap two entity names in the response."""
        if len(entities) < 2:
            return response, "swap_entity_names"

        # Find two entities that appear in the response
        present_entities = [e for e in entities if e.get("name", "") in response]
        if len(present_entities) < 2:
            # Use any two entities
            present_entities = entities[:2]

        e1, e2 = self.rng.sample(present_entities, 2)
        name1, name2 = e1.get("name", "Entity1"), e2.get("name", "Entity2")

        # Swap names
        placeholder = "___PLACEHOLDER___"
        perturbed = response.replace(name1, placeholder)
        perturbed = perturbed.replace(name2, name1)
        perturbed = perturbed.replace(placeholder, name2)

        return perturbed, "swap_entity_names"

    def invent_relationship(
        self,
        response: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Add a fabricated relationship to the response."""
        fake_relationships = [
            "founded",
            "co-authored a paper with",
            "was a student of",
            "collaborated extensively with",
            "worked together at",
            "jointly developed",
            "was influenced by the work of",
            "received funding from",
        ]

        if len(entities) < 2:
            return response, "invent_relationship"

        e1, e2 = self.rng.sample(entities[:5], 2)
        name1, name2 = e1.get("name", "Entity1"), e2.get("name", "Entity2")
        rel = self.rng.choice(fake_relationships)

        # Add fake relationship claim
        fake_claim = f" Additionally, {name1} {rel} {name2}."
        perturbed = response.rstrip() + fake_claim

        return perturbed, "invent_relationship"

    def wrong_attribute(
        self,
        response: str,
        context: str,
    ) -> Tuple[str, str]:
        """Change dates, numbers, or other attributes incorrectly."""
        # Find years in the response
        years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', response)
        if years:
            year = self.rng.choice(years)
            # Change year by +/- 5-20 years
            delta = self.rng.choice([-20, -15, -10, -5, 5, 10, 15, 20])
            wrong_year = str(int(year) + delta)
            perturbed = response.replace(year, wrong_year, 1)
            return perturbed, "wrong_attribute"

        # Find numbers
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            num = self.rng.choice(numbers)
            # Modify number
            try:
                n = int(num)
                if n > 0:
                    wrong_num = str(n * self.rng.choice([2, 3, 5]) if n < 100 else n + self.rng.randint(50, 200))
                    perturbed = response.replace(num, wrong_num, 1)
                    return perturbed, "wrong_attribute"
            except ValueError:
                pass

        # Fallback: change descriptive terms
        replacements = [
            ("significant", "minor"),
            ("major", "minor"),
            ("important", "trivial"),
            ("successful", "failed"),
            ("first", "last"),
            ("early", "late"),
            ("before", "after"),
        ]

        for orig, repl in replacements:
            if orig in response.lower():
                perturbed = re.sub(rf'\b{orig}\b', repl, response, count=1, flags=re.IGNORECASE)
                return perturbed, "wrong_attribute"

        return response, "wrong_attribute"

    def mix_entities(
        self,
        response: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Mix facts from different entities together."""
        if len(entities) < 2:
            return response, "mix_entities"

        # Find entity mentioned in response
        main_entity = None
        other_entity = None
        for e in entities:
            name = e.get("name", "")
            if name and name in response:
                if main_entity is None:
                    main_entity = e
                else:
                    other_entity = e
                    break

        if other_entity is None and len(entities) >= 2:
            other_entity = entities[1] if entities[0] == main_entity else entities[0]

        if main_entity and other_entity:
            other_desc = other_entity.get("description", "")
            if other_desc:
                # Add info from other entity as if it belongs to main entity
                main_name = main_entity.get("name", "it")
                fake_info = f" {main_name} is also known for: {other_desc[:100]}..."
                perturbed = response.rstrip() + fake_info
                return perturbed, "mix_entities"

        return response, "mix_entities"

    def add_unsupported_claim(
        self,
        response: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Add claims not supported by the context."""
        unsupported_claims = [
            "This had a profound impact on subsequent developments in the field.",
            "Many historians consider this to be one of the most important events of the era.",
            "The implications of this are still being studied today.",
            "This approach was later adopted worldwide.",
            "Critics at the time were initially skeptical, but it proved to be correct.",
            "This marked a turning point in the field.",
            "The discovery led to several Nobel Prize nominations.",
            "This fundamentally changed how experts think about the subject.",
        ]

        claim = self.rng.choice(unsupported_claims)
        perturbed = response.rstrip() + " " + claim

        return perturbed, "add_unsupported_claim"

    def contradict_context(
        self,
        response: str,
        context: str,
    ) -> Tuple[str, str]:
        """Add a statement that contradicts the context."""
        # Try to find a negatable statement
        patterns = [
            (r"(.+) is (.+)", lambda m: f" However, some sources dispute that {m.group(1)} is {m.group(2)}."),
            (r"(.+) was (.+)", lambda m: f" Though it's worth noting that {m.group(1)} was not actually {m.group(2)} according to some accounts."),
            (r"(.+) developed (.+)", lambda m: f" Some historians argue that {m.group(1)} did not actually develop {m.group(2)}."),
        ]

        for pattern, generator in patterns:
            match = re.search(pattern, response[:200])
            if match:
                contradiction = generator(match)
                perturbed = response.rstrip() + contradiction
                return perturbed, "contradict_context"

        # Fallback: generic contradiction
        perturbed = response.rstrip() + " However, this account is disputed by some sources."
        return perturbed, "contradict_context"

    def perturb(
        self,
        response: str,
        context: str,
        entities: List[Dict[str, Any]],
        perturbation_type: Optional[PerturbationType] = None,
    ) -> Tuple[str, str]:
        """
        Apply a perturbation to create an unfaithful response.

        Args:
            response: Original faithful response
            context: The context the response is based on
            entities: List of entities in the context
            perturbation_type: Specific perturbation to apply, or None for random

        Returns:
            Tuple of (perturbed_response, perturbation_type_used)
        """
        if perturbation_type is None:
            perturbation_type = self.rng.choice([
                "swap_entity_names",
                "invent_relationship",
                "wrong_attribute",
                "mix_entities",
                "add_unsupported_claim",
                "contradict_context",
            ])

        if perturbation_type == "swap_entity_names":
            return self.swap_entity_names(response, entities)
        elif perturbation_type == "invent_relationship":
            return self.invent_relationship(response, entities)
        elif perturbation_type == "wrong_attribute":
            return self.wrong_attribute(response, context)
        elif perturbation_type == "mix_entities":
            return self.mix_entities(response, entities)
        elif perturbation_type == "add_unsupported_claim":
            return self.add_unsupported_claim(response, entities)
        elif perturbation_type == "contradict_context":
            return self.contradict_context(response, context)
        else:
            return self.add_unsupported_claim(response, entities)


class DPODataGenerator:
    """
    Generate DPO training data for faithfulness.

    Two approaches:
    1. LLM hallucinations: Ask teacher LLM to generate plausible but incorrect answers
    2. Perturbations: Systematically modify correct answers
    """

    def __init__(
        self,
        teacher_llm=None,
        seed: int = 42,
    ):
        self.teacher_llm = teacher_llm
        self.rng = random.Random(seed)
        self.perturbation_gen = PerturbationGenerator(seed)
        self._sample_counter = 0

    def _generate_sample_id(self, method: str) -> str:
        """Generate a unique sample ID."""
        self._sample_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dpo_{method}_{timestamp}_{self._sample_counter}"

    def generate_from_perturbation(
        self,
        context: str,
        question: str,
        faithful_response: str,
        entities: List[Dict[str, Any]],
        perturbation_type: Optional[PerturbationType] = None,
    ) -> DPOSample:
        """
        Generate DPO sample using perturbation.

        Args:
            context: Retrieved context
            question: User question
            faithful_response: Correct, grounded response
            entities: Entities in the context
            perturbation_type: Specific perturbation to use

        Returns:
            DPOSample with chosen and rejected responses
        """
        # Generate rejected response through perturbation
        rejected, pert_type = self.perturbation_gen.perturb(
            response=faithful_response,
            context=context,
            entities=entities,
            perturbation_type=perturbation_type,
        )

        # Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        return DPOSample(
            id=self._generate_sample_id("perturbation"),
            prompt=prompt,
            chosen=faithful_response,
            rejected=rejected,
            perturbation_type=pert_type,
            metadata={
                "method": "perturbation",
                "num_entities": len(entities),
            },
        )

    def generate_from_llm_hallucination(
        self,
        context: str,
        question: str,
        faithful_response: str,
    ) -> Optional[DPOSample]:
        """
        Generate DPO sample using LLM-generated hallucination.

        Args:
            context: Retrieved context
            question: User question
            faithful_response: Correct, grounded response

        Returns:
            DPOSample or None if LLM not available
        """
        if self.teacher_llm is None:
            logger.warning("Teacher LLM not configured, cannot generate hallucinations")
            return None

        # Prompt LLM to generate plausible but incorrect answer
        hallucination_prompt = f"""You are generating training data for detecting unfaithful AI responses.

Given the following context and question, generate a response that:
1. Sounds plausible and confident
2. Contains at least one factual error or unsupported claim
3. May include information NOT in the context
4. Should NOT be obviously wrong

Context:
{context}

Question: {question}

Correct answer (for reference, do NOT just copy this):
{faithful_response}

Generate an INCORRECT but plausible-sounding response:"""

        try:
            # Call teacher LLM
            hallucinated = self.teacher_llm.invoke(hallucination_prompt)
            if hasattr(hallucinated, 'content'):
                hallucinated = hallucinated.content

            # Build prompt
            prompt = f"Context:\n{context}\n\nQuestion: {question}"

            return DPOSample(
                id=self._generate_sample_id("llm_hallucination"),
                prompt=prompt,
                chosen=faithful_response,
                rejected=hallucinated,
                perturbation_type="llm_hallucination",
                metadata={
                    "method": "llm_hallucination",
                },
            )

        except Exception as e:
            logger.warning(f"LLM hallucination generation failed: {e}")
            return None

    def generate_from_tool_conversation(
        self,
        conversation: Dict[str, Any],
        use_llm_hallucinations: bool = False,
    ) -> Optional[DPOSample]:
        """
        Generate DPO sample from a tool-use conversation.

        Args:
            conversation: Tool conversation dict with turns
            use_llm_hallucinations: Whether to use LLM for rejected response

        Returns:
            DPOSample or None if generation fails
        """
        turns = conversation.get("turns", [])
        if len(turns) < 4:  # Need at least: user, assistant+tool_call, tool_response, assistant_answer
            return None

        # Extract question (first user turn)
        question = None
        for turn in turns:
            if turn.get("role") == "user":
                question = turn.get("content", "")
                break

        if not question:
            return None

        # Extract context from tool responses
        context_parts = []
        for turn in turns:
            if turn.get("role") == "tool":
                tool_content = turn.get("content", "")
                if isinstance(tool_content, str):
                    try:
                        tool_content = json.loads(tool_content)
                    except json.JSONDecodeError:
                        pass
                if isinstance(tool_content, dict):
                    # Extract entity info
                    entity = tool_content.get("entity", {})
                    if entity:
                        context_parts.append(
                            f"{entity.get('name', 'Unknown')} ({entity.get('type', '')}): "
                            f"{entity.get('description', 'No description')}"
                        )
                    # Extract relationships
                    rels = tool_content.get("relationships", [])
                    for rel in rels[:5]:
                        if isinstance(rel, dict):
                            context_parts.append(
                                f"- {rel.get('type', 'RELATED_TO')}: {rel.get('target', 'Unknown')}"
                            )
                else:
                    context_parts.append(str(tool_content)[:500])

        if not context_parts:
            return None

        context = "\n".join(context_parts)

        # Extract faithful response (last assistant turn with content)
        faithful_response = None
        for turn in reversed(turns):
            if turn.get("role") == "assistant" and turn.get("content") and not turn.get("tool_calls"):
                faithful_response = turn.get("content", "")
                break

        if not faithful_response:
            return None

        # Extract entities for perturbation
        entities = []
        for turn in turns:
            if turn.get("role") == "tool":
                tool_content = turn.get("content", "")
                if isinstance(tool_content, str):
                    try:
                        tool_content = json.loads(tool_content)
                    except json.JSONDecodeError:
                        continue
                if isinstance(tool_content, dict):
                    entity = tool_content.get("entity")
                    if entity:
                        entities.append(entity)
                    # Also get from relationships
                    for rel in tool_content.get("relationships", []):
                        if isinstance(rel, dict) and rel.get("target"):
                            entities.append({
                                "name": rel.get("target"),
                                "type": rel.get("target_type", "Entity"),
                            })

        # Generate DPO sample
        if use_llm_hallucinations and self.teacher_llm:
            sample = self.generate_from_llm_hallucination(context, question, faithful_response)
            if sample:
                return sample

        # Fallback to perturbation
        return self.generate_from_perturbation(
            context=context,
            question=question,
            faithful_response=faithful_response,
            entities=entities,
        )

    def generate_dataset(
        self,
        conversations: List[Dict[str, Any]],
        use_llm_hallucinations: bool = False,
        llm_ratio: float = 0.3,
    ) -> List[DPOSample]:
        """
        Generate DPO dataset from tool-use conversations.

        Args:
            conversations: List of tool-use conversation dicts
            use_llm_hallucinations: Whether to use LLM for some rejected responses
            llm_ratio: Ratio of samples to generate with LLM (if enabled)

        Returns:
            List of DPOSample objects
        """
        samples = []

        for conv in conversations:
            # Decide whether to use LLM for this sample
            use_llm = use_llm_hallucinations and self.rng.random() < llm_ratio

            sample = self.generate_from_tool_conversation(conv, use_llm_hallucinations=use_llm)
            if sample:
                samples.append(sample)

        logger.info(f"Generated {len(samples)}/{len(conversations)} DPO samples")

        # Log perturbation type distribution
        type_counts = {}
        for s in samples:
            pt = s.perturbation_type or "unknown"
            type_counts[pt] = type_counts.get(pt, 0) + 1
        logger.info(f"Perturbation distribution: {type_counts}")

        return samples
