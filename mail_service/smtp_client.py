"""
SMTP email client for sending approval requests and notifications.
"""

import os
import ssl
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailClient:
    """
    SMTP email client for sending emails.

    Configuration via environment variables:
    - SMTP_HOST: SMTP server host
    - SMTP_PORT: SMTP server port (default: 587)
    - SMTP_USER: SMTP username
    - SMTP_PASSWORD: SMTP password
    - EMAIL_FROM: From email address
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_from: Optional[str] = None,
        use_tls: bool = True,
    ):
        """
        Initialize email client.

        Args:
            smtp_host: SMTP server host.
            smtp_port: SMTP server port.
            smtp_user: SMTP username.
            smtp_password: SMTP password.
            email_from: From email address.
            use_tls: Whether to use TLS.
        """
        self.smtp_host = smtp_host or os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.environ.get("SMTP_USER", "")
        self.smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD", "")
        self.email_from = email_from or os.environ.get("EMAIL_FROM", self.smtp_user)
        self.use_tls = use_tls

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    def send_email(
        self,
        to: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        cc: Optional[List[str]] = None,
    ) -> bool:
        """
        Send an email.

        Args:
            to: Recipient email address.
            subject: Email subject.
            body_text: Plain text body.
            body_html: Optional HTML body.
            cc: Optional CC addresses.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            logger.error("Email not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = to

            if cc:
                msg["Cc"] = ", ".join(cc)

            # Add text part
            msg.attach(MIMEText(body_text, "plain"))

            # Add HTML part if provided
            if body_html:
                msg.attach(MIMEText(body_html, "html"))

            # Send email
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)

                recipients = [to]
                if cc:
                    recipients.extend(cc)

                server.sendmail(self.email_from, recipients, msg.as_string())

            logger.info(f"Email sent to {to}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_approval_request(
        self,
        to: str,
        topics_text: str,
        session_id: str,
    ) -> bool:
        """
        Send a research topic approval request.

        Args:
            to: Recipient email address.
            topics_text: Formatted topics text.
            session_id: Approval session ID.

        Returns:
            True if sent successfully.
        """
        subject = "Knowledge Graph Research - Topic Approval Request"

        body_text = f"""
Knowledge Graph Research Topics

Please review the following research topics and reply with the
numbers you want to approve (e.g., 1,2,4,7,9,10):

{topics_text}

Session ID: {session_id}

To approve, simply reply to this email with the topic numbers
you want to research, separated by commas.

Example reply: 1,2,4,7,9,10
"""

        body_html = f"""
<html>
<body>
<h2>Knowledge Graph Research Topics</h2>

<p>Please review the following research topics and reply with the
numbers you want to approve (e.g., 1,2,4,7,9,10):</p>

<pre style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
{topics_text}
</pre>

<p><strong>Session ID:</strong> {session_id}</p>

<p>To approve, simply reply to this email with the topic numbers
you want to research, separated by commas.</p>

<p><em>Example reply: 1,2,4,7,9,10</em></p>
</body>
</html>
"""

        return self.send_email(
            to=to,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
        )

    def send_research_summary(
        self,
        to: str,
        summary: dict,
    ) -> bool:
        """
        Send a research completion summary.

        Args:
            to: Recipient email address.
            summary: Research summary dict.

        Returns:
            True if sent successfully.
        """
        subject = "Knowledge Graph Research - Completed"

        topics_completed = summary.get("topics_completed", 0)
        documents_added = summary.get("documents_added", 0)
        entities_created = summary.get("entities_created", 0)
        duration = summary.get("duration_minutes", 0)

        body_text = f"""
Knowledge Graph Research Complete

Summary:
- Topics researched: {topics_completed}
- Documents added: {documents_added}
- Entities created: {entities_created}
- Duration: {duration} minutes

The knowledge graph has been updated with the new information.
"""

        body_html = f"""
<html>
<body>
<h2>Knowledge Graph Research Complete</h2>

<table style="border-collapse: collapse; margin: 20px 0;">
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><strong>Topics Researched</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{topics_completed}</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><strong>Documents Added</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{documents_added}</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><strong>Entities Created</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{entities_created}</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><strong>Duration</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{duration} minutes</td>
</tr>
</table>

<p>The knowledge graph has been updated with the new information.</p>
</body>
</html>
"""

        return self.send_email(
            to=to,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
        )
