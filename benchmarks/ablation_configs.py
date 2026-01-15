#!/usr/bin/env python3
"""
Shared Ablation Study Configurations.

Contains unified config dataclass and config sets for:
- STUDY1_CONFIGS: Follow-up planning variants (from followup_ablation_study.py)
- IMPROVED_CONFIGS: Hybrid RAG variants (from improved_ablation_study.py)
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class AblationConfig:
    """Unified configuration for ablation studies.

    Supports both follow-up planning mode and standard hybrid RAG mode.
    """
    name: str
    description: str
    # Planning mode flags
    use_retrieval_planning: bool = False  # Old CLaRa-style planning
    use_followup_planning: bool = False   # New follow-up question planning
    planning_reasoning: bool = False       # Detailed thinking for follow-up
    # Primary search params (for followup mode)
    primary_vector_limit: int = 5
    primary_max_hops: int = 4
    # Secondary search params (for followup mode)
    secondary_vector_limit: int = 3
    secondary_max_hops: int = 2
    # Standard params (used when not in followup mode)
    max_hops: int = 2
    vector_limit: int = 5
    # Feature flags
    compression_enabled: bool = True
    web_search_enabled: bool = False
    auto_add_documents: bool = True
    use_improved_retrieval: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# STUDY1_CONFIGS: Follow-Up Planning Variants
# From followup_ablation_study.py
# =============================================================================

STUDY1_CONFIGS: List[AblationConfig] = [
    # Baseline: no_planning (previous best performer)
    AblationConfig(
        name="no_planning_baseline",
        description="No planning, direct vector search (previous best)",
        use_retrieval_planning=False,
        use_followup_planning=False,
        max_hops=2,
        vector_limit=5,
    ),

    # Old planning style for comparison
    AblationConfig(
        name="old_planning",
        description="Old CLaRa-style entity/relationship planning",
        use_retrieval_planning=True,
        use_followup_planning=False,
        max_hops=2,
        vector_limit=5,
    ),

    # Follow-up planning: v4_h4 primary
    AblationConfig(
        name="followup_v4h4",
        description="Follow-up planning: v4_h4 primary + v3_h2 secondary",
        use_followup_planning=True,
        primary_vector_limit=4,
        primary_max_hops=4,
        secondary_vector_limit=3,
        secondary_max_hops=2,
    ),

    # Follow-up planning: v5_h5 primary
    AblationConfig(
        name="followup_v5h5",
        description="Follow-up planning: v5_h5 primary + v3_h2 secondary",
        use_followup_planning=True,
        primary_vector_limit=5,
        primary_max_hops=5,
        secondary_vector_limit=3,
        secondary_max_hops=2,
    ),

    # Follow-up planning: v4_h6 primary
    AblationConfig(
        name="followup_v4h6",
        description="Follow-up planning: v4_h6 primary + v3_h2 secondary",
        use_followup_planning=True,
        primary_vector_limit=4,
        primary_max_hops=6,
        secondary_vector_limit=3,
        secondary_max_hops=2,
    ),

    # Follow-up planning with reasoning: v4_h4
    AblationConfig(
        name="followup_reasoning_v4h4",
        description="Follow-up + reasoning: v4_h4 primary + v3_h2 secondary",
        use_followup_planning=True,
        planning_reasoning=True,
        primary_vector_limit=4,
        primary_max_hops=4,
        secondary_vector_limit=3,
        secondary_max_hops=2,
    ),

    # Follow-up planning with reasoning: v5_h5
    AblationConfig(
        name="followup_reasoning_v5h5",
        description="Follow-up + reasoning: v5_h5 primary + v3_h2 secondary",
        use_followup_planning=True,
        planning_reasoning=True,
        primary_vector_limit=5,
        primary_max_hops=5,
        secondary_vector_limit=3,
        secondary_max_hops=2,
    ),
]


# =============================================================================
# IMPROVED_CONFIGS: Hybrid RAG Variants
# From improved_ablation_study.py (converted from dict to AblationConfig list)
# =============================================================================

IMPROVED_CONFIGS: List[AblationConfig] = [
    # === Feature Flag Tests (vec=5, hop=2, all tools ON) ===
    AblationConfig(
        name="baseline",
        description="All features ON (default)",
        use_improved_retrieval=True,
        vector_limit=5,
        max_hops=2,
        compression_enabled=True,
        web_search_enabled=True,
        use_retrieval_planning=True,
        auto_add_documents=True,
    ),
    AblationConfig(
        name="no_planning",
        description="Disable retrieval planning",
        use_improved_retrieval=True,
        vector_limit=5,
        max_hops=2,
        compression_enabled=True,
        web_search_enabled=True,
        use_retrieval_planning=False,
        auto_add_documents=True,
    ),
    AblationConfig(
        name="no_compression",
        description="Disable context compression",
        use_improved_retrieval=True,
        vector_limit=5,
        max_hops=2,
        compression_enabled=False,
        web_search_enabled=True,
        use_retrieval_planning=True,
        auto_add_documents=True,
    ),

    # === Vec/Hop Grid Search (all tools ON) ===
    AblationConfig(
        name="v5_h3",
        description="Best performer: vec=5, hop=3",
        use_improved_retrieval=True,
        vector_limit=5,
        max_hops=3,
        compression_enabled=True,
        web_search_enabled=True,
        use_retrieval_planning=True,
        auto_add_documents=True,
    ),
    AblationConfig(
        name="v4_h4",
        description="Deeper traversal: vec=4, hop=4",
        use_improved_retrieval=True,
        vector_limit=4,
        max_hops=4,
        compression_enabled=True,
        web_search_enabled=True,
        use_retrieval_planning=True,
        auto_add_documents=True,
    ),
    AblationConfig(
        name="v7_h3",
        description="Wider deep: vec=7, hop=3",
        use_improved_retrieval=True,
        vector_limit=7,
        max_hops=3,
        compression_enabled=True,
        web_search_enabled=True,
        use_retrieval_planning=True,
        auto_add_documents=True,
    ),

    # === Edge Case ===
    AblationConfig(
        name="minimal",
        description="All features OFF (graph only)",
        use_improved_retrieval=True,
        vector_limit=3,
        max_hops=1,
        compression_enabled=False,
        web_search_enabled=False,
        use_retrieval_planning=False,
        auto_add_documents=False,
    ),
]


def generate_study2_configs(best_primary_vec: int, best_primary_hops: int, use_reasoning: bool) -> List[AblationConfig]:
    """Generate Study 2 configs based on best primary settings from Study 1."""
    secondary_variations = [
        (2, 1),  # v2_h1 - minimal
        (3, 2),  # v3_h2 - default
        (3, 3),  # v3_h3 - deeper
        (4, 2),  # v4_h2 - wider
        (4, 3),  # v4_h3 - wider + deeper
        (5, 2),  # v5_h2 - very wide
    ]

    configs = []
    for sec_vec, sec_hops in secondary_variations:
        configs.append(AblationConfig(
            name=f"best_primary_sec_v{sec_vec}h{sec_hops}",
            description=f"Best primary (v{best_primary_vec}_h{best_primary_hops}) + secondary v{sec_vec}_h{sec_hops}",
            use_followup_planning=True,
            planning_reasoning=use_reasoning,
            primary_vector_limit=best_primary_vec,
            primary_max_hops=best_primary_hops,
            secondary_vector_limit=sec_vec,
            secondary_max_hops=sec_hops,
        ))

    return configs
