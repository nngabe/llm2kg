"""
RAGAS Helper Module.

Configures RAGAS to use Ollama embeddings and multi-provider LLM with fallback.
Uses the same LLM configuration as LLM-as-Judge (Gemini 2.5 Pro primary, GPT-5.2 fallback).
"""

import os
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Cache for initialized wrappers
_cached_llm = None
_cached_embeddings = None


def get_ollama_host() -> str:
    """Get Ollama host from environment."""
    return os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


def _get_llm_with_fallback(
    primary_provider: str = "google",
    primary_model: str = "gemini-2.5-pro",
    fallback_provider: str = "openai",
    fallback_model: str = "gpt-5-mini",
    temperature: float = 0.0,
) -> Any:
    """
    Get LLM with automatic fallback support.

    Uses the same provider logic as LLMJudgeMetric.

    Args:
        primary_provider: Primary LLM provider (google, openai, anthropic)
        primary_model: Primary model name
        fallback_provider: Fallback LLM provider
        fallback_model: Fallback model name
        temperature: Temperature for LLM responses

    Returns:
        LangChain LLM instance
    """
    def _create_llm(provider: str, model: str):
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Try primary provider
    try:
        llm = _create_llm(primary_provider, primary_model)
        logger.info(f"RAGAS LLM initialized with {primary_provider}/{primary_model}")
        return llm
    except Exception as e:
        logger.warning(f"Primary LLM ({primary_provider}/{primary_model}) failed: {e}")

    # Try fallback provider
    try:
        llm = _create_llm(fallback_provider, fallback_model)
        logger.info(f"RAGAS LLM fallback to {fallback_provider}/{fallback_model}")
        return llm
    except Exception as e:
        logger.error(f"Fallback LLM ({fallback_provider}/{fallback_model}) also failed: {e}")
        raise


def get_ragas_config(
    embedding_model: str = "qwen3-embedding:8b",
    llm_provider: str = "google",
    llm_model: str = "gemini-2.5-pro",
    fallback_provider: str = "openai",
    fallback_model: str = "gpt-5-mini",
) -> Tuple[Any, Any]:
    """
    Get RAGAS-compatible LLM and embeddings wrappers.

    Uses Ollama for embeddings (qwen3-embedding:8b) and multi-provider LLM
    with the same primary/fallback as LLM-as-Judge.

    Args:
        embedding_model: Ollama embedding model name
        llm_provider: Primary LLM provider (google, openai, anthropic)
        llm_model: Primary LLM model name
        fallback_provider: Fallback LLM provider
        fallback_model: Fallback LLM model name

    Returns:
        Tuple of (llm_wrapper, embeddings_wrapper) for RAGAS evaluate()
    """
    global _cached_llm, _cached_embeddings

    if _cached_llm is not None and _cached_embeddings is not None:
        return _cached_llm, _cached_embeddings

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import OllamaEmbeddings

        ollama_host = get_ollama_host()

        # Initialize Ollama embeddings (qwen3-embedding:8b)
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_host,
        )

        # Initialize LLM with fallback (same as LLM-as-Judge)
        llm = _get_llm_with_fallback(
            primary_provider=llm_provider,
            primary_model=llm_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

        # Wrap for RAGAS
        _cached_embeddings = LangchainEmbeddingsWrapper(embeddings)
        _cached_llm = LangchainLLMWrapper(llm)

        logger.info(f"RAGAS configured: {embedding_model} embeddings, {llm_provider}/{llm_model} LLM")
        return _cached_llm, _cached_embeddings

    except ImportError as e:
        logger.error(f"Failed to import RAGAS/Langchain components: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to initialize RAGAS config: {e}")
        return None, None


def run_ragas_evaluate(
    data,
    metrics: list,
    embedding_model: str = "qwen3-embedding:8b",
    llm_provider: str = "google",
    llm_model: str = "gemini-2.5-pro",
    fallback_provider: str = "openai",
    fallback_model: str = "gpt-5-mini",
) -> Optional[dict]:
    """
    Run RAGAS evaluate with Ollama embeddings and multi-provider LLM.

    Args:
        data: Either a HuggingFace Dataset OR a list of dicts with keys:
              user_input, response, retrieved_contexts (and optionally reference)
        metrics: List of RAGAS metrics to evaluate
        embedding_model: Ollama embedding model name (default: qwen3-embedding:8b)
        llm_provider: Primary LLM provider (default: google)
        llm_model: Primary LLM model (default: gemini-2.5-pro)
        fallback_provider: Fallback LLM provider (default: openai)
        fallback_model: Fallback LLM model (default: gpt-5-mini)

    Returns:
        Evaluation result dict or None on failure
    """
    try:
        from ragas import evaluate
        from ragas import EvaluationDataset, SingleTurnSample

        llm, embeddings = get_ragas_config(
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

        if llm is None or embeddings is None:
            logger.warning("RAGAS config not available, skipping evaluation")
            return None

        # Convert list of dicts to EvaluationDataset if needed
        if isinstance(data, list):
            samples = []
            for item in data:
                sample = SingleTurnSample(
                    user_input=item.get("user_input", item.get("question", "")),
                    response=item.get("response", item.get("answer", "")),
                    retrieved_contexts=item.get("retrieved_contexts", item.get("contexts", [])),
                    reference=item.get("reference", item.get("ground_truth", None)),
                )
                samples.append(sample)
            dataset = EvaluationDataset(samples=samples)
        else:
            dataset = data

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )

        return result

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return None


def extract_score(result: Any, metric_name: str) -> Optional[float]:
    """
    Safely extract a score from RAGAS evaluation result.

    Handles both scalar and list returns from RAGAS.

    Args:
        result: RAGAS evaluation result
        metric_name: Name of the metric to extract

    Returns:
        Float score or None if extraction fails
    """
    if result is None:
        return None

    try:
        # Result might be an EvaluationResult object or dict
        if hasattr(result, '__getitem__'):
            value = result[metric_name]
        elif hasattr(result, 'scores'):
            value = result.scores.get(metric_name)
        else:
            logger.warning(f"Unknown result type: {type(result)}")
            return None

        # Handle list vs scalar
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            # Average if multiple values
            return sum(float(v) for v in value) / len(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            logger.warning(f"Unexpected value type for {metric_name}: {type(value)}")
            return None

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Failed to extract score for {metric_name}: {e}")
        return None


def clear_cache():
    """Clear cached LLM and embeddings wrappers."""
    global _cached_llm, _cached_embeddings
    _cached_llm = None
    _cached_embeddings = None
