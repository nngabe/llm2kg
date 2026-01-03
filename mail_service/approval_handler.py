"""
Approval handler for research topic approval workflow.

Manages approval sessions with token-based verification.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval session."""
    PENDING = "pending"
    APPROVED = "approved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalSession(BaseModel):
    """An approval session for research topics."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = Field(default_factory=lambda: (datetime.now() + timedelta(hours=24)).isoformat())
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    topic_numbers: List[int] = Field(default_factory=list)
    approved_numbers: List[int] = Field(default_factory=list)
    user_email: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalHandler:
    """
    Handler for research topic approvals.

    Supports both email and in-app approval workflows.
    Sessions are stored in a JSON file for persistence.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        expiry_hours: int = 24,
    ):
        """
        Initialize the approval handler.

        Args:
            storage_path: Path to store approval sessions.
            expiry_hours: Hours until session expires.
        """
        self.expiry_hours = expiry_hours

        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self.storage_path = data_dir / "approval_sessions.json"

        self._sessions: Dict[str, ApprovalSession] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for session_id, session_data in data.items():
                        self._sessions[session_id] = ApprovalSession(**session_data)
            except Exception as e:
                logger.warning(f"Failed to load sessions: {e}")

    def _save_sessions(self):
        """Save sessions to storage."""
        try:
            data = {
                sid: session.model_dump()
                for sid, session in self._sessions.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def create_session(
        self,
        topic_numbers: List[int],
        user_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalSession:
        """
        Create a new approval session.

        Args:
            topic_numbers: Available topic numbers.
            user_email: Optional user email.
            metadata: Optional metadata.

        Returns:
            New approval session.
        """
        session = ApprovalSession(
            topic_numbers=topic_numbers,
            user_email=user_email,
            metadata=metadata or {},
            expires_at=(datetime.now() + timedelta(hours=self.expiry_hours)).isoformat(),
        )

        self._sessions[session.session_id] = session
        self._save_sessions()

        logger.info(f"Created approval session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ApprovalSession]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)

        if session:
            # Check expiry
            if datetime.now() > datetime.fromisoformat(session.expires_at):
                session.status = ApprovalStatus.EXPIRED
                self._save_sessions()

        return session

    def process_approval(
        self,
        session_id: str,
        approved_numbers: List[int],
    ) -> Optional[ApprovalSession]:
        """
        Process an approval response.

        Args:
            session_id: Session ID.
            approved_numbers: Numbers of approved topics.

        Returns:
            Updated session or None if invalid.
        """
        session = self.get_session(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        if session.status != ApprovalStatus.PENDING:
            logger.warning(f"Session not pending: {session_id} ({session.status})")
            return session

        # Validate approved numbers
        valid_numbers = [
            n for n in approved_numbers
            if n in session.topic_numbers
        ]

        session.approved_numbers = valid_numbers
        session.status = ApprovalStatus.APPROVED

        self._save_sessions()

        logger.info(f"Approved topics for session {session_id}: {valid_numbers}")
        return session

    def cancel_session(self, session_id: str) -> bool:
        """Cancel an approval session."""
        session = self.get_session(session_id)

        if not session:
            return False

        session.status = ApprovalStatus.CANCELLED
        self._save_sessions()

        logger.info(f"Cancelled session: {session_id}")
        return True

    def check_status(self, session_id: str) -> ApprovalStatus:
        """Check the status of a session."""
        session = self.get_session(session_id)
        if session:
            return session.status
        return ApprovalStatus.EXPIRED

    def get_approved_topics(self, session_id: str) -> List[int]:
        """Get approved topic numbers for a session."""
        session = self.get_session(session_id)
        if session and session.status == ApprovalStatus.APPROVED:
            return session.approved_numbers
        return []

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = datetime.now()
        expired = []

        for sid, session in self._sessions.items():
            if datetime.fromisoformat(session.expires_at) < now:
                expired.append(sid)

        for sid in expired:
            del self._sessions[sid]

        if expired:
            self._save_sessions()
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def parse_approval_reply(self, reply_text: str) -> List[int]:
        """
        Parse an approval reply to extract topic numbers.

        Handles formats like:
        - "1,2,4,7,9,10"
        - "1, 2, 4, 7, 9, 10"
        - "approve 1 2 4 7"

        Args:
            reply_text: Reply text to parse.

        Returns:
            List of approved topic numbers.
        """
        import re

        # Extract all numbers from the reply
        numbers = re.findall(r'\d+', reply_text)

        # Convert to integers and filter to valid range
        approved = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= 10:
                    approved.append(num)
            except ValueError:
                continue

        return sorted(set(approved))
