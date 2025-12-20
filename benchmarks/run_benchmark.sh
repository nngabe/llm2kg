#!/bin/bash
# Run the model comparison benchmark
#
# Usage:
#   ./run_benchmark.sh                    # Run all models (requires Ollama)
#   ./run_benchmark.sh --skip_local       # Skip Ollama models (API only)
#   ./run_benchmark.sh --subject law      # Use law dataset
#

cd "$(dirname "$0")/.."

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set - OpenAI models will fail"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set - Claude models will fail"
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Warning: GOOGLE_API_KEY not set - Gemini models will fail"
    echo "  To configure: export GOOGLE_API_KEY=your-api-key"
    echo "  Get a key at: https://aistudio.google.com/apikey"
fi

# Run benchmark
python -m benchmarks.model_comparison "$@"
