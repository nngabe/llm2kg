#!/usr/bin/env python3
"""
Reranker server for zerank-2 cross-encoder model.

Run on Mac host:
    pip install sentence-transformers flask
    python reranker_server.py

The server exposes a /rerank endpoint that accepts query-document pairs
and returns relevance scores using the zerank-2 cross-encoder model.
"""

import argparse
import logging
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and tokenizer instances
model = None
tokenizer = None
yes_token_id = None


# Chat template for zerank-2 (query-document relevance)
RERANK_TEMPLATE = """<|im_start|>user
Document: {document}

Query: {query}

Is this document relevant to the query? Answer only Yes or No.<|im_end|>
<|im_start|>assistant
"""


def load_model(model_name: str = "zeroentropy/zerank-2"):
    """Load the zerank-2 model as a causal LM."""
    global model, tokenizer, yes_token_id
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get "Yes" token ID for scoring
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    logger.info(f"Yes token ID: {yes_token_id}")

    # Move to GPU if available
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    device = str(next(model.parameters()).device) if model is not None else "none"
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "yes_token_id": yes_token_id,
    })


@app.route("/rerank", methods=["POST"])
def rerank():
    """
    Rerank documents against a query.

    Request JSON:
    {
        "query": "What is the efficiency of LM2500?",
        "documents": [
            {"id": "doc1", "content": "The LM2500 has 38% efficiency..."},
            {"id": "doc2", "content": "Gas turbines are machines..."}
        ]
    }

    Response JSON:
    {
        "scores": [
            {"id": "doc1", "score": 0.92},
            {"id": "doc2", "score": 0.15}
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    query = data.get("query", "")
    documents = data.get("documents", [])

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not documents:
        return jsonify({"error": "No documents provided"}), 400

    try:
        doc_ids = []
        scores = []
        device = next(model.parameters()).device

        # Process each document individually
        for doc in documents:
            content = doc.get("content", "")[:2000]  # Truncate long documents
            doc_id = doc.get("id", str(len(doc_ids)))
            doc_ids.append(doc_id)

            # Format using chat template
            prompt = RERANK_TEMPLATE.format(query=query, document=content)

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(device)

            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                # Get logits at the last position
                last_logits = outputs.logits[0, -1, :]
                # Get "Yes" token logit and apply sigmoid with scaling (from zerank code)
                yes_logit = last_logits[yes_token_id].float()
                score = torch.sigmoid(yes_logit / 5.0).item()
                scores.append(score)

        # Build response
        results = [
            {"id": doc_id, "score": float(score)}
            for doc_id, score in zip(doc_ids, scores)
        ]

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Reranked {len(documents)} documents for query: {query[:50]}...")

        return jsonify({"scores": results})

    except Exception as e:
        logger.error(f"Rerank error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rerank/batch", methods=["POST"])
def rerank_batch():
    """
    Batch rerank multiple queries (optional optimization endpoint).

    Request JSON:
    {
        "requests": [
            {"query": "...", "documents": [...]},
            {"query": "...", "documents": [...]}
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    requests_list = data.get("requests", [])
    device = next(model.parameters()).device

    all_results = []
    for req in requests_list:
        query = req.get("query", "")
        documents = req.get("documents", [])

        doc_ids = []
        scores = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")[:2000]
            doc_ids.append(doc.get("id", str(i)))

            prompt = RERANK_TEMPLATE.format(query=query, document=content)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                yes_logit = last_logits[yes_token_id].float()
                score = torch.sigmoid(yes_logit / 5.0).item()
                scores.append(score)

        req_results = [
            {"id": doc_id, "score": float(score)}
            for doc_id, score in zip(doc_ids, scores)
        ]
        req_results.sort(key=lambda x: x["score"], reverse=True)
        all_results.append({"scores": req_results})

    return jsonify({"results": all_results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zerank-2 reranker server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on")
    parser.add_argument("--model", default="zeroentropy/zerank-2", help="Model to load")
    args = parser.parse_args()

    load_model(args.model)

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
