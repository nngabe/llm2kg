"""
Research Mode Handler for Chainlit.

Handles autonomous research to fill knowledge gaps:
- Gap detection in knowledge graph
- Topic generation and prioritization
- Email/in-app approval workflow
- Autonomous research with graceful termination
- Progress tracking
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set
from enum import Enum

import chainlit as cl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchStatus(str, Enum):
    """Status of research operation."""
    IDLE = "idle"
    DETECTING_GAPS = "detecting_gaps"
    GENERATING_TOPICS = "generating_topics"
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


class ResearchTopic:
    """A research topic identified for exploration."""

    def __init__(
        self,
        number: int,
        title: str,
        description: str,
        priority: float = 0.5,
        estimated_documents: int = 5,
    ):
        self.number = number
        self.title = title
        self.description = description
        self.priority = priority
        self.estimated_documents = estimated_documents
        self.approved = False
        self.completed = False
        self.documents_added = 0
        self.entities_created = 0


class ResearchModeHandler:
    """
    Handler for Research mode in Chainlit.

    Features:
    - Knowledge gap detection
    - Topic generation (numbered 1-10)
    - Email + in-app approval workflow
    - Autonomous research with time limits
    - Progress tracking and graceful termination
    """

    def __init__(
        self,
        default_time_limit_minutes: int = 30,
        max_topics: int = 10,
        require_approval: bool = True,
    ):
        """
        Initialize Research mode handler.

        Args:
            default_time_limit_minutes: Default time limit for research.
            max_topics: Maximum number of topics to generate.
            require_approval: Whether to require topic approval.
        """
        self.default_time_limit = default_time_limit_minutes
        self.max_topics = max_topics
        self.require_approval = require_approval

        # State
        self.status = ResearchStatus.IDLE
        self.topics: List[ResearchTopic] = []
        self.approved_topics: Set[int] = set()
        self.start_time: Optional[datetime] = None
        self.time_limit_minutes: int = default_time_limit_minutes
        self.should_terminate = False

        # Stats
        self.total_documents_added = 0
        self.total_entities_created = 0

        # Components
        self._gap_detector = None
        self._topic_generator = None
        self._research_agent = None
        self._email_client = None

    async def initialize(self) -> bool:
        """
        Initialize research components.

        Returns:
            True if initialization successful.
        """
        try:
            # These will be implemented in the research package
            # For now, we'll use placeholder implementations
            logger.info("Research mode initialized (components pending)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Research mode: {e}")
            return False

    async def handle_message(self, message: cl.Message) -> None:
        """
        Handle a user message in Research mode.

        Commands:
        - "start research" or "find gaps": Start gap detection
        - "approve 1,2,4,7": Approve specific topics
        - "approve all": Approve all topics
        - "stop" or "terminate": Stop research gracefully
        - "status": Show current status
        - "set time X": Set time limit to X minutes
        """
        content = message.content.lower().strip()

        # Parse commands
        if content in ["start research", "find gaps", "detect gaps"]:
            await self._start_gap_detection()

        elif content.startswith("approve "):
            args = content[8:].strip()
            if args == "all":
                await self._approve_all_topics()
            else:
                # Parse comma-separated numbers
                try:
                    numbers = [int(x.strip()) for x in args.split(",")]
                    await self._approve_topics(numbers)
                except ValueError:
                    await cl.Message(
                        content="Invalid format. Use: `approve 1,2,4,7` or `approve all`",
                        author="System",
                    ).send()

        elif content in ["stop", "terminate", "cancel"]:
            await self._terminate_research()

        elif content == "status":
            await self._show_status()

        elif content.startswith("set time "):
            try:
                minutes = int(content[9:].strip())
                self.time_limit_minutes = min(max(minutes, 5), 120)
                await cl.Message(
                    content=f"Time limit set to {self.time_limit_minutes} minutes.",
                    author="System",
                ).send()
            except ValueError:
                await cl.Message(
                    content="Invalid time. Use: `set time 30`",
                    author="System",
                ).send()

        elif content.startswith("email "):
            # Email approval request
            email = content[6:].strip()
            await self._send_approval_email(email)

        else:
            # Show help
            await self._show_help()

    async def _start_gap_detection(self) -> None:
        """Start gap detection process."""
        self.status = ResearchStatus.DETECTING_GAPS

        async with cl.Step(name="Gap Detection", type="tool") as step:
            step.input = "Analyzing knowledge graph for gaps"

            await cl.Message(
                content="Analyzing knowledge graph for gaps...",
                author="System",
            ).send()

            # Simulate gap detection (actual implementation in research package)
            await asyncio.sleep(1)  # Placeholder

            # Generate topics
            self.status = ResearchStatus.GENERATING_TOPICS
            step.output = "Gap detection complete"

        await self._generate_topics()

    async def _generate_topics(self) -> None:
        """Generate research topics based on detected gaps."""
        async with cl.Step(name="Topic Generation", type="llm") as step:
            step.input = "Generating research topics"

            # Placeholder topics (actual implementation in research package)
            self.topics = [
                ResearchTopic(
                    number=i,
                    title=f"Research Topic {i}",
                    description=f"Description for topic {i}",
                    priority=1.0 - (i * 0.1),
                )
                for i in range(1, min(self.max_topics + 1, 11))
            ]

            step.output = f"Generated {len(self.topics)} topics"

        # Display topics for approval
        await self._display_topics_for_approval()

    async def _display_topics_for_approval(self) -> None:
        """Display topics and request approval."""
        self.status = ResearchStatus.AWAITING_APPROVAL

        content_parts = [
            "**Research Topics Identified:**\n",
            "Reply with `approve 1,2,4,7` to approve specific topics,",
            "or `approve all` to approve all topics.\n",
        ]

        for topic in self.topics:
            priority_bar = "█" * int(topic.priority * 5) + "░" * (5 - int(topic.priority * 5))
            content_parts.append(
                f"**{topic.number}.** {topic.title}\n"
                f"   Priority: {priority_bar} | Est. docs: {topic.estimated_documents}\n"
                f"   _{topic.description}_\n"
            )

        content_parts.append(
            f"\nTime limit: **{self.time_limit_minutes} minutes** "
            f"(change with `set time X`)"
        )

        await cl.Message(
            content="\n".join(content_parts),
            author="Research Assistant",
        ).send()

        # Create approval actions
        actions = [
            cl.Action(
                name="approve_all",
                value="all",
                label="Approve All",
            ),
            cl.Action(
                name="cancel_research",
                value="cancel",
                label="Cancel",
            ),
        ]

        await cl.Message(
            content="Select an action or type your approval:",
            author="System",
            actions=actions,
        ).send()

    async def _approve_topics(self, numbers: List[int]) -> None:
        """Approve specific topics by number."""
        valid_numbers = [n for n in numbers if 1 <= n <= len(self.topics)]
        self.approved_topics = set(valid_numbers)

        for topic in self.topics:
            topic.approved = topic.number in self.approved_topics

        approved_titles = [
            self.topics[n - 1].title
            for n in sorted(self.approved_topics)
        ]

        await cl.Message(
            content=(
                f"Approved {len(self.approved_topics)} topics:\n"
                + "\n".join(f"- {title}" for title in approved_titles)
            ),
            author="System",
        ).send()

        if self.approved_topics:
            await self._start_research()

    async def _approve_all_topics(self) -> None:
        """Approve all topics."""
        self.approved_topics = set(range(1, len(self.topics) + 1))
        for topic in self.topics:
            topic.approved = True

        await cl.Message(
            content=f"All {len(self.topics)} topics approved!",
            author="System",
        ).send()

        await self._start_research()

    async def _start_research(self) -> None:
        """Start the autonomous research process."""
        self.status = ResearchStatus.RESEARCHING
        self.start_time = datetime.now()
        self.should_terminate = False

        await cl.Message(
            content=(
                f"Starting research on {len(self.approved_topics)} topics.\n"
                f"Time limit: {self.time_limit_minutes} minutes.\n"
                f"Use `stop` to terminate gracefully."
            ),
            author="Research Assistant",
        ).send()

        # Run research with progress updates
        await self._research_loop()

    async def _research_loop(self) -> None:
        """Main research loop with progress updates."""
        end_time = self.start_time + timedelta(minutes=self.time_limit_minutes)

        for topic in self.topics:
            if not topic.approved:
                continue

            if self.should_terminate:
                break

            if datetime.now() >= end_time:
                await cl.Message(
                    content="Time limit reached. Terminating gracefully...",
                    author="System",
                ).send()
                break

            # Research this topic
            async with cl.Step(name=f"Researching: {topic.title}", type="run") as step:
                step.input = topic.description

                await cl.Message(
                    content=f"Researching topic {topic.number}: **{topic.title}**",
                    author="Research Assistant",
                ).send()

                # Placeholder research (actual implementation in research package)
                await asyncio.sleep(2)  # Simulate research

                # Simulate adding documents
                topic.documents_added = 3
                topic.entities_created = 5
                topic.completed = True

                self.total_documents_added += topic.documents_added
                self.total_entities_created += topic.entities_created

                step.output = (
                    f"Added {topic.documents_added} documents, "
                    f"created {topic.entities_created} entities"
                )

                # Progress update
                completed = sum(1 for t in self.topics if t.completed)
                total = len([t for t in self.topics if t.approved])
                await cl.Message(
                    content=f"Progress: {completed}/{total} topics completed",
                    author="System",
                ).send()

        # Research completed
        self.status = ResearchStatus.COMPLETED if not self.should_terminate else ResearchStatus.TERMINATED
        await self._show_summary()

    async def _terminate_research(self) -> None:
        """Terminate research gracefully."""
        self.should_terminate = True

        if self.status == ResearchStatus.RESEARCHING:
            await cl.Message(
                content="Termination requested. Finishing current topic...",
                author="System",
            ).send()
        else:
            self.status = ResearchStatus.TERMINATED
            await cl.Message(
                content="Research cancelled.",
                author="System",
            ).send()

    async def _show_status(self) -> None:
        """Show current research status."""
        parts = [f"**Status:** {self.status.value}"]

        if self.topics:
            approved = sum(1 for t in self.topics if t.approved)
            completed = sum(1 for t in self.topics if t.completed)
            parts.append(f"**Topics:** {completed}/{approved} completed")

        if self.start_time:
            elapsed = datetime.now() - self.start_time
            remaining = timedelta(minutes=self.time_limit_minutes) - elapsed
            parts.append(f"**Time remaining:** {max(0, int(remaining.total_seconds() // 60))} minutes")

        parts.append(f"**Documents added:** {self.total_documents_added}")
        parts.append(f"**Entities created:** {self.total_entities_created}")

        await cl.Message(
            content="\n".join(parts),
            author="System",
        ).send()

    async def _show_summary(self) -> None:
        """Show research summary."""
        completed_topics = [t for t in self.topics if t.completed]

        content = [
            "## Research Summary\n",
            f"**Status:** {'Completed' if self.status == ResearchStatus.COMPLETED else 'Terminated'}",
            f"**Topics researched:** {len(completed_topics)}",
            f"**Documents added:** {self.total_documents_added}",
            f"**Entities created:** {self.total_entities_created}",
            "",
            "### Completed Topics:",
        ]

        for topic in completed_topics:
            content.append(
                f"- **{topic.title}**: "
                f"{topic.documents_added} docs, {topic.entities_created} entities"
            )

        await cl.Message(
            content="\n".join(content),
            author="Research Assistant",
        ).send()

    async def _send_approval_email(self, email: str) -> None:
        """Send approval email with numbered topics."""
        if not self.topics:
            await cl.Message(
                content="No topics to send. Run gap detection first.",
                author="System",
            ).send()
            return

        # Placeholder for email functionality
        # Actual implementation in email package

        topic_list = "\n".join(
            f"{t.number}. {t.title}: {t.description}"
            for t in self.topics
        )

        await cl.Message(
            content=(
                f"Email would be sent to: **{email}**\n\n"
                f"Topics:\n{topic_list}\n\n"
                f"Reply with numbers to approve (e.g., 1,2,4,7,9,10)"
            ),
            author="System",
        ).send()

    async def _show_help(self) -> None:
        """Show help for research mode."""
        await cl.Message(
            content="""**Research Mode Commands:**

- `start research` - Detect gaps and generate topics
- `approve 1,2,4,7` - Approve specific topics
- `approve all` - Approve all topics
- `stop` - Terminate research gracefully
- `status` - Show current status
- `set time 30` - Set time limit (minutes)
- `email user@example.com` - Send approval request via email

**Workflow:**
1. Run `start research` to detect gaps
2. Review generated topics (numbered 1-10)
3. Approve topics you want to research
4. Research runs autonomously with progress updates
5. Use `stop` to terminate gracefully at any time
""",
            author="System",
        ).send()

    async def close(self) -> None:
        """Clean up resources."""
        self.should_terminate = True
        self.status = ResearchStatus.IDLE
