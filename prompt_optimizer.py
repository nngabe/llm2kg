"""
Prompt Optimizer using NeMo-style Genetic Algorithm with Mutation Operators.

This module implements prompt optimization inspired by NVIDIA NeMo-Agent-Toolkit's
approach, using a genetic algorithm with LLM-powered mutations to evolve prompts.

Mutation Operators:
- tighten: Remove redundancies and verbosity
- reorder: Optimize logical sequence of instructions
- constrain: Add explicit rules and boundaries
- harden: Enhance error handling
- defuse: Replace vague language with measurable actions
- format_lock: Enforce JSON/XML schemas for structured output
"""

import os
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b")


@dataclass
class PromptOptimizationConfig:
    """Configuration for genetic algorithm prompt optimization."""
    ga_population_size: int = 16
    ga_generations: int = 8
    ga_mutation_rate: float = 0.2
    ga_crossover_rate: float = 0.7
    ga_elitism: int = 2
    ga_tournament_size: int = 3
    ga_parallel_evaluations: int = 4
    ga_diversity_lambda: float = 0.1  # Diversity penalty to prevent convergence


@dataclass
class PromptCandidate:
    """A candidate prompt in the population."""
    text: str
    fitness: float = 0.0
    generation: int = 0
    parent_operators: List[str] = field(default_factory=list)


# Meta-prompt for applying mutation operators
MUTATION_META_PROMPT = """You are a prompt engineering expert. Your task is to improve the given prompt using a specific mutation operator.

## Original Prompt:
{original_prompt}

## Objective:
{objective}

## Mutation Operator: {operator}

{operator_instructions}

## Rules:
1. Return ONLY the improved prompt text
2. Do not include explanations or commentary
3. Preserve the core functionality and intent
4. Do not use curly braces except for template variables like {{variable}}

Improved prompt:"""


OPERATOR_INSTRUCTIONS = {
    "tighten": """Remove redundancies and verbosity while preserving meaning.
- Eliminate repeated instructions
- Combine similar guidelines
- Remove filler words and phrases
- Make each sentence count""",

    "reorder": """Optimize the logical sequence of instructions.
- Put most important instructions first
- Group related instructions together
- Ensure logical flow from context to action
- Place constraints near their related instructions""",

    "constrain": """Add explicit rules and boundaries.
- Add specific output format requirements
- Define edge case handling
- Add validation criteria
- Specify what NOT to do""",

    "harden": """Enhance error handling and robustness.
- Add fallback behaviors
- Specify what to do when information is missing
- Add recovery strategies for failures
- Include timeout/retry guidance""",

    "defuse": """Replace vague language with measurable actions.
- 'Be careful' -> 'Verify X before Y'
- 'Be thorough' -> 'Include at least N items'
- 'Be honest' -> 'State confidence as percentage'
- Make all instructions actionable""",

    "format_lock": """Enforce strict output schemas.
- Add JSON schema requirements
- Specify required fields
- Add type constraints
- Include example outputs""",
}


class PromptOptimizer:
    """
    Optimize prompts using genetic algorithm with LLM-powered mutations.

    Implements NeMo-style prompt optimization with:
    - Population-based evolution over multiple generations
    - Six mutation operators for intelligent modifications
    - Tournament selection for parent selection
    - Elitism to preserve top performers
    """

    MUTATION_OPERATORS = [
        "tighten",
        "reorder",
        "constrain",
        "harden",
        "defuse",
        "format_lock",
    ]

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[PromptOptimizationConfig] = None,
    ):
        """
        Initialize the prompt optimizer.

        Args:
            llm: Language model for mutations. Defaults to Ollama Nemotron.
            config: GA configuration. Defaults to PromptOptimizationConfig().
        """
        self.llm = llm or ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.7,  # Higher temp for creative mutations
        )
        self.config = config or PromptOptimizationConfig()
        self._generation = 0

    def optimize(
        self,
        prompt: str,
        objective: str,
        eval_fn: Callable[[str], float],
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evolve prompt population to maximize eval_fn score.

        Args:
            prompt: Original prompt to optimize
            objective: Description of what the prompt should achieve
            eval_fn: Function that scores a prompt (0-1, higher is better)
            verbose: Print progress information

        Returns:
            Tuple of (best_prompt, optimization_stats)
        """
        # Initialize population with variations of original prompt
        population = self._initialize_population(prompt, objective)

        # Evaluate initial population
        population = self._evaluate_population(population, eval_fn)

        best_ever = max(population, key=lambda x: x.fitness)
        stats = {
            "generations": [],
            "best_fitness_history": [],
            "operators_used": {op: 0 for op in self.MUTATION_OPERATORS},
        }

        for gen in range(self.config.ga_generations):
            self._generation = gen + 1

            if verbose:
                best = max(population, key=lambda x: x.fitness)
                avg = sum(p.fitness for p in population) / len(population)
                logger.info(f"Generation {gen + 1}: best={best.fitness:.3f}, avg={avg:.3f}")

            # Selection and reproduction
            new_population = []

            # Elitism: preserve top performers
            elite = sorted(population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(elite[:self.config.ga_elitism])

            # Fill rest with offspring
            while len(new_population) < self.config.ga_population_size:
                # Tournament selection
                parent = self._tournament_select(population)

                # Apply mutation or crossover
                if random.random() < self.config.ga_mutation_rate:
                    operator = random.choice(self.MUTATION_OPERATORS)
                    child_text = self.mutate(parent.text, operator, objective)
                    stats["operators_used"][operator] += 1
                    child = PromptCandidate(
                        text=child_text,
                        generation=self._generation,
                        parent_operators=parent.parent_operators + [operator],
                    )
                else:
                    # Keep parent with small variation
                    child = PromptCandidate(
                        text=parent.text,
                        generation=self._generation,
                        parent_operators=parent.parent_operators.copy(),
                    )

                new_population.append(child)

            # Evaluate new population
            population = self._evaluate_population(new_population, eval_fn)

            # Track best
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best

            stats["generations"].append({
                "generation": gen + 1,
                "best_fitness": gen_best.fitness,
                "avg_fitness": sum(p.fitness for p in population) / len(population),
                "diversity": self._calculate_diversity(population),
            })
            stats["best_fitness_history"].append(gen_best.fitness)

        stats["final_best"] = {
            "fitness": best_ever.fitness,
            "generation": best_ever.generation,
            "operators_applied": best_ever.parent_operators,
        }

        return best_ever.text, stats

    def mutate(self, prompt: str, operator: str, objective: str) -> str:
        """
        Apply a mutation operator to the prompt.

        Args:
            prompt: Prompt to mutate
            operator: One of MUTATION_OPERATORS
            objective: Prompt's intended purpose

        Returns:
            Mutated prompt text
        """
        if operator not in self.MUTATION_OPERATORS:
            raise ValueError(f"Unknown operator: {operator}. Must be one of {self.MUTATION_OPERATORS}")

        mutation_prompt = MUTATION_META_PROMPT.format(
            original_prompt=prompt,
            objective=objective,
            operator=operator,
            operator_instructions=OPERATOR_INSTRUCTIONS[operator],
        )

        try:
            response = self.llm.invoke([
                HumanMessage(content=mutation_prompt),
            ])
            mutated = response.content.strip()

            # Basic validation
            if len(mutated) < 50:  # Too short, probably failed
                logger.warning(f"Mutation {operator} produced short output, keeping original")
                return prompt

            return mutated

        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            return prompt

    def apply_all_operators(self, prompt: str, objective: str) -> Dict[str, str]:
        """
        Apply all mutation operators and return results.

        Useful for exploring how each operator modifies a prompt.

        Args:
            prompt: Original prompt
            objective: Prompt's intended purpose

        Returns:
            Dict mapping operator name to mutated prompt
        """
        results = {}
        for operator in self.MUTATION_OPERATORS:
            results[operator] = self.mutate(prompt, operator, objective)
        return results

    def _initialize_population(
        self,
        prompt: str,
        objective: str,
    ) -> List[PromptCandidate]:
        """Initialize population with variations of original prompt."""
        population = [
            PromptCandidate(text=prompt, generation=0)  # Original
        ]

        # Create variations using each operator
        for operator in self.MUTATION_OPERATORS:
            if len(population) >= self.config.ga_population_size:
                break
            mutated = self.mutate(prompt, operator, objective)
            population.append(PromptCandidate(
                text=mutated,
                generation=0,
                parent_operators=[operator],
            ))

        # Fill remaining slots with random mutations
        while len(population) < self.config.ga_population_size:
            operator = random.choice(self.MUTATION_OPERATORS)
            base = random.choice(population[:len(self.MUTATION_OPERATORS) + 1])
            mutated = self.mutate(base.text, operator, objective)
            population.append(PromptCandidate(
                text=mutated,
                generation=0,
                parent_operators=base.parent_operators + [operator],
            ))

        return population

    def _evaluate_population(
        self,
        population: List[PromptCandidate],
        eval_fn: Callable[[str], float],
    ) -> List[PromptCandidate]:
        """Evaluate fitness of all candidates in population."""
        # Parallel evaluation for efficiency
        with ThreadPoolExecutor(max_workers=self.config.ga_parallel_evaluations) as executor:
            futures = {
                executor.submit(eval_fn, candidate.text): candidate
                for candidate in population
            }

            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    candidate.fitness = future.result()
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    candidate.fitness = 0.0

        return population

    def _tournament_select(self, population: List[PromptCandidate]) -> PromptCandidate:
        """Select a parent using tournament selection."""
        tournament = random.sample(population, min(self.config.ga_tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _calculate_diversity(self, population: List[PromptCandidate]) -> float:
        """Calculate diversity score for population (0-1)."""
        if len(population) <= 1:
            return 0.0

        # Simple diversity: count unique prompts
        unique = len(set(p.text for p in population))
        return unique / len(population)


def create_eval_function_from_dataset(
    dataset: List[Dict[str, Any]],
    agent_class: Any,
    agent_kwargs: Dict[str, Any],
) -> Callable[[str], float]:
    """
    Create an evaluation function from a Q&A dataset.

    Args:
        dataset: List of {"question": str, "expected_answer": str}
        agent_class: Agent class to instantiate
        agent_kwargs: Additional kwargs for agent

    Returns:
        Function that scores a prompt based on agent performance
    """
    def eval_fn(prompt: str) -> float:
        # Update system prompt
        kwargs = agent_kwargs.copy()
        kwargs["system_prompt"] = prompt

        agent = agent_class(**kwargs)
        scores = []

        for item in dataset:
            try:
                response = agent.answer_question(item["question"])
                # Simple similarity scoring (could use better metrics)
                expected = item["expected_answer"].lower()
                actual = response.answer.lower()

                # Word overlap score
                expected_words = set(expected.split())
                actual_words = set(actual.split())
                if expected_words:
                    overlap = len(expected_words & actual_words) / len(expected_words)
                    scores.append(overlap)
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                scores.append(0.0)

        agent.close()
        return sum(scores) / len(scores) if scores else 0.0

    return eval_fn


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Optimizer CLI")
    parser.add_argument("--prompt", type=str, help="Prompt to optimize")
    parser.add_argument("--prompt-file", type=str, help="File containing prompt")
    parser.add_argument("--objective", type=str, required=True, help="Optimization objective")
    parser.add_argument("--operator", type=str, help="Apply single operator")
    parser.add_argument("--all-operators", action="store_true", help="Apply all operators")
    parser.add_argument("--generations", type=int, default=8, help="Number of GA generations")
    parser.add_argument("--population", type=int, default=16, help="Population size")
    parser.add_argument("--output", type=str, help="Output file for results")
    args = parser.parse_args()

    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Must provide --prompt or --prompt-file")
        exit(1)

    config = PromptOptimizationConfig(
        ga_generations=args.generations,
        ga_population_size=args.population,
    )
    optimizer = PromptOptimizer(config=config)

    if args.operator:
        # Apply single operator
        result = optimizer.mutate(prompt, args.operator, args.objective)
        print(f"\n=== {args.operator.upper()} ===")
        print(result)

    elif args.all_operators:
        # Apply all operators
        results = optimizer.apply_all_operators(prompt, args.objective)
        for op, mutated in results.items():
            print(f"\n=== {op.upper()} ===")
            print(mutated)
            print()

    else:
        print("Use --operator <name> or --all-operators to apply mutations")
        print(f"Available operators: {optimizer.MUTATION_OPERATORS}")
