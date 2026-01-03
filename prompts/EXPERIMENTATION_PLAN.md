# System Prompt Experimentation Plan

## Current Issues Identified

### 1. Empty Responses for Short Documents
- **Problem:** LLM returns empty responses for documents with ~400-900 chars when using the full prompt
- **Root Cause:** Prompt length (7004 chars) >> Input length ratio causes model confusion/timeout
- **Evidence:** Same documents work with minimal prompts (50 chars)

### 2. Relationship Type Explosion (52 vs target 5-20)
- **Problem:** Too many specific relationship types created
- **Examples of overly specific types:**
  - `EXHIBITS_INVERSE_TRADEOFF_WITH`
  - `DO_NOT_DETERMINE`
  - `HAS_DECISION_MAKING`
  - `FACES_PRESSURE_FROM`
- **Root Cause:** Prompt encourages descriptive types but doesn't constrain to canonical set

### 3. Inconsistent Naming Conventions
- **Problem:** Mix of good (`IS_A`, `INFLUENCES`) and verbose (`CAUSED_DESTRUCTION_OF`) types
- **Root Cause:** Examples in prompt may be too varied

---

## Experimentation Variables

### A. Prompt Length Variants

| Variant | Description | Chars | Use Case |
|---------|-------------|-------|----------|
| `MINIMAL` | No examples, just schema | ~500 | Baseline |
| `COMPACT` | 1 example, essential rules | ~2000 | Short docs |
| `STANDARD` | 2 examples, full rules | ~4000 | Medium docs |
| `DETAILED` | 3 examples, edge cases | ~7000 | Complex docs |

### B. Relationship Type Constraints

| Approach | Description |
|----------|-------------|
| Open | Current: any SCREAMING_SNAKE_CASE |
| Suggested | Provide list of ~20 preferred types |
| Constrained | Must use from predefined set of 15 |
| Hierarchical | Core types + allowed extensions |

### C. Description Length Control

| Setting | Description |
|---------|-------------|
| Brief | 5-10 words |
| Standard | 10-20 words (current) |
| Detailed | 20-40 words |

---

## Proposed Prompt Variants

### Variant 1: COMPACT (for short documents < 1000 chars)

```
You are a knowledge graph extraction expert.

RULES:
- Node IDs: Use canonical names (not pronouns)
- Node Labels: PascalCase (e.g., Person, Organization, Concept)
- Relationship Types: Use these preferred types when applicable:
  IS_A, PART_OF, RELATES_TO, INFLUENCES, CAUSES, CREATES,
  LOCATED_IN, MEMBER_OF, WORKS_FOR, OWNS, USES, PRODUCES

OUTPUT: Valid JSON with "nodes" and "relationships" arrays.
Each node: {{"id": "...", "labels": ["..."], "properties": {{"description": "..."}}}}
Each relationship: {{"source": "...", "target": "...", "type": "...", "properties": {{"description": "..."}}}}
```

### Variant 2: STANDARD (current, for docs 1000-3000 chars)

Keep current prompt but add relationship type guidance.

### Variant 3: ADAPTIVE (select based on input length)

```python
def get_prompt(text_length: int) -> str:
    if text_length < 800:
        return KG_EXTRACTION_PROMPT_COMPACT
    elif text_length < 2500:
        return KG_EXTRACTION_PROMPT_STANDARD
    else:
        return KG_EXTRACTION_PROMPT_DETAILED
```

---

## Experiment Design

### Experiment 1: Prompt Length Impact
- **Hypothesis:** Shorter prompts improve success rate for short documents
- **Method:** Test 20 documents with each prompt variant
- **Metrics:** Success rate, nodes/doc, relationships/doc, type diversity

### Experiment 2: Relationship Constraint Impact
- **Hypothesis:** Constrained relationship types reduce ontology explosion
- **Method:** Compare open vs suggested vs constrained approaches
- **Metrics:** Unique relationship types, type frequency distribution

### Experiment 3: Adaptive Prompt Selection
- **Hypothesis:** Matching prompt length to document length optimizes extraction
- **Method:** Run 50 docs with adaptive selection
- **Metrics:** Overall success rate, extraction quality

---

## Implementation Plan

### Phase 1: Create Prompt Variants (Day 1)
1. Create `KG_EXTRACTION_PROMPT_MINIMAL` - bare bones, ~500 chars
2. Create `KG_EXTRACTION_PROMPT_COMPACT` - 1 example, ~2000 chars
3. Modify `KG_EXTRACTION_PROMPT` to include relationship suggestions

### Phase 2: Add Adaptive Selection (Day 1)
1. Add `get_extraction_prompt(text_length)` function
2. Modify `_extract_node()` to use adaptive selection
3. Add logging to track which prompt was used

### Phase 3: Run Experiments (Day 2)
1. Test each variant on 20-document sample
2. Collect metrics
3. Analyze results

### Phase 4: Optimize (Day 3)
1. Select best performing approach
2. Fine-tune thresholds and prompt content
3. Run full dataset test

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Success Rate | 80% | 95%+ |
| Relationship Types | 52 | 15-25 |
| Empty Responses | 20% | <5% |
| Avg Nodes/Doc | 11.6 | 10-15 |
| Avg Edges/Doc | 11.5 | 8-12 |

---

## Files to Create/Modify

1. `/app/prompts/agent_skb_prompts.py` - Add new prompt variants
2. `/app/prompts/__init__.py` - Export new prompts
3. `/app/agent_skb.py` - Add adaptive prompt selection
4. `/app/scripts/test_prompts.py` - Experiment runner script
