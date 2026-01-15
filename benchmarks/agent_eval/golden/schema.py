"""
Golden Dataset Schema.

Defines the structure for golden test cases and datasets.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from ..metrics.base import TestCase

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase(TestCase):
    """Extended test case with generation and review metadata.

    Inherits all fields from TestCase and adds:
    - Generation metadata (model, timestamp, prompt)
    - Review status and notes
    - Validation scores
    """

    # Generation metadata
    generated_by: str = "unknown"  # Model that generated this
    generated_at: Optional[str] = None
    generation_prompt: Optional[str] = None

    # Review metadata
    reviewed: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_status: Literal["pending", "approved", "rejected", "needs_revision"] = "pending"

    # Validation
    validation_score: Optional[float] = None
    validation_notes: List[str] = field(default_factory=list)

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    def approve(self, reviewer: str, notes: Optional[str] = None) -> None:
        """Mark this test case as approved."""
        self.reviewed = True
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.now().isoformat()
        self.review_status = "approved"
        if notes:
            self.reviewer_notes = notes

    def reject(self, reviewer: str, reason: str) -> None:
        """Mark this test case as rejected."""
        self.reviewed = True
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.now().isoformat()
        self.review_status = "rejected"
        self.reviewer_notes = reason

    def request_revision(self, reviewer: str, notes: str) -> None:
        """Mark this test case as needing revision."""
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.now().isoformat()
        self.review_status = "needs_revision"
        self.reviewer_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including all metadata."""
        base = super().to_dict()
        base.update({
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "generation_prompt": self.generation_prompt,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "review_status": self.review_status,
            "validation_score": self.validation_score,
            "validation_notes": self.validation_notes,
            "tags": self.tags,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenTestCase":
        """Create GoldenTestCase from dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            question=data.get("question", ""),
            expected_answer=data.get("expected_answer"),
            expected_entities=data.get("expected_entities", []),
            expected_relationships=data.get("expected_relationships", []),
            ground_truth_context=data.get("ground_truth_context", []),
            optimal_tool_sequence=data.get("optimal_tool_sequence", []),
            minimum_steps=data.get("minimum_steps"),
            should_reject=data.get("should_reject", False),
            rejection_reason=data.get("rejection_reason"),
            type=data.get("type", "all"),
            difficulty=data.get("difficulty", "medium"),
            source=data.get("source", "generated"),
            reviewed=data.get("reviewed", False),
            reviewer_notes=data.get("reviewer_notes"),
            metadata=data.get("metadata", {}),
            generated_by=data.get("generated_by", "unknown"),
            generated_at=data.get("generated_at"),
            generation_prompt=data.get("generation_prompt"),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            review_status=data.get("review_status", "pending"),
            validation_score=data.get("validation_score"),
            validation_notes=data.get("validation_notes", []),
            tags=data.get("tags", []),
        )


@dataclass
class GoldenDataset:
    """Collection of golden test cases with management utilities."""

    name: str
    version: str = "1.0"
    description: str = ""
    test_cases: List[GoldenTestCase] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test_case(self, test_case: GoldenTestCase) -> None:
        """Add a test case to the dataset."""
        self.test_cases.append(test_case)
        self.updated_at = datetime.now().isoformat()

    def remove_test_case(self, test_case_id: str) -> bool:
        """Remove a test case by ID."""
        for i, tc in enumerate(self.test_cases):
            if tc.id == test_case_id:
                self.test_cases.pop(i)
                self.updated_at = datetime.now().isoformat()
                return True
        return False

    def get_test_case(self, test_case_id: str) -> Optional[GoldenTestCase]:
        """Get a test case by ID."""
        for tc in self.test_cases:
            if tc.id == test_case_id:
                return tc
        return None

    def filter_by_type(self, test_type: str) -> List[GoldenTestCase]:
        """Filter test cases by type."""
        return [tc for tc in self.test_cases if tc.type == test_type or tc.type == "all"]

    def filter_by_status(self, status: str) -> List[GoldenTestCase]:
        """Filter test cases by review status."""
        return [tc for tc in self.test_cases if tc.review_status == status]

    def filter_by_difficulty(self, difficulty: str) -> List[GoldenTestCase]:
        """Filter test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]

    def filter_by_tags(self, tags: List[str]) -> List[GoldenTestCase]:
        """Filter test cases that have any of the given tags."""
        return [tc for tc in self.test_cases if any(t in tc.tags for t in tags)]

    def get_approved(self) -> List[GoldenTestCase]:
        """Get only approved test cases."""
        return self.filter_by_status("approved")

    def get_pending_review(self) -> List[GoldenTestCase]:
        """Get test cases pending review."""
        return self.filter_by_status("pending")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total = len(self.test_cases)
        if total == 0:
            return {"total": 0}

        by_type = {}
        by_status = {}
        by_difficulty = {}
        by_source = {}

        for tc in self.test_cases:
            by_type[tc.type] = by_type.get(tc.type, 0) + 1
            by_status[tc.review_status] = by_status.get(tc.review_status, 0) + 1
            by_difficulty[tc.difficulty] = by_difficulty.get(tc.difficulty, 0) + 1
            by_source[tc.source] = by_source.get(tc.source, 0) + 1

        return {
            "total": total,
            "by_type": by_type,
            "by_status": by_status,
            "by_difficulty": by_difficulty,
            "by_source": by_source,
            "approved_count": by_status.get("approved", 0),
            "pending_count": by_status.get("pending", 0),
            "rejection_test_count": sum(1 for tc in self.test_cases if tc.should_reject),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenDataset":
        """Create GoldenDataset from dictionary."""
        dataset = cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )

        for tc_data in data.get("test_cases", []):
            dataset.add_test_case(GoldenTestCase.from_dict(tc_data))

        return dataset

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved dataset to {path}")

    @classmethod
    def load(cls, path: Path) -> "GoldenDataset":
        """Load dataset from JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        dataset = cls.from_dict(data)
        logger.info(f"Loaded dataset from {path}: {len(dataset.test_cases)} test cases")
        return dataset

    def merge(self, other: "GoldenDataset") -> "GoldenDataset":
        """Merge another dataset into this one."""
        existing_ids = {tc.id for tc in self.test_cases}

        for tc in other.test_cases:
            if tc.id not in existing_ids:
                self.add_test_case(tc)
            else:
                logger.warning(f"Skipping duplicate test case: {tc.id}")

        self.updated_at = datetime.now().isoformat()
        return self
