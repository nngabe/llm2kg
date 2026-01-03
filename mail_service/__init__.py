"""
Mail service package for approval workflow.

Note: This package is named 'mail_service' instead of 'email' to avoid
conflicts with Python's built-in email module.

Components:
- EmailClient: SMTP-based email sending
- ApprovalHandler: Token-based approval processing
"""

from .smtp_client import EmailClient
from .approval_handler import ApprovalHandler, ApprovalSession

__all__ = [
    "EmailClient",
    "ApprovalHandler",
    "ApprovalSession",
]
