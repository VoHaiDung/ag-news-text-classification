"""
Email Notifier Implementation for AG News Text Classification
================================================================================
This module implements email notification delivery using SMTP protocol with
support for HTML templates, attachments, and delivery tracking.

The email notifier provides:
- SMTP and SMTP/TLS support
- HTML and plain text emails
- Attachment handling
- Batch email delivery

References:
    - RFC 5321: Simple Mail Transfer Protocol
    - Python email package documentation
    - Email best practices for deliverability

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any
import re
from pathlib import Path

from src.services.notification.notification_service import NotificationProvider, Notification
from src.utils.logging_config import get_logger


class EmailNotifier(NotificationProvider):
    """
    Email notification provider using SMTP.
    
    This provider handles email delivery with support for various
    SMTP servers and email formats.
    """
    
    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        use_tls: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender_email: Optional[str] = None,
        sender_name: str = "AG News Classification System",
        max_recipients_per_message: int = 50
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            use_tls: Use TLS encryption
            username: SMTP authentication username
            password: SMTP authentication password
            sender_email: Sender email address
            sender_name: Sender display name
            max_recipients_per_message: Maximum recipients per email
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.use_tls = use_tls
        self.username = username
        self.password = password
        self.sender_email = sender_email or username
        self.sender_name = sender_name
        self.max_recipients_per_message = max_recipients_per_message
        
        # Email validation pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # Connection pool
        self._smtp_connection = None
        self._connection_lock = asyncio.Lock()
        
        self.logger = get_logger("notifier.email")
    
    async def send(self, notification: Notification) -> bool:
        """
        Send email notification.
        
        Args:
            notification: Notification to send
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Validate recipient
            if not await self.validate_recipient(notification.recipient):
                self.logger.error(f"Invalid email recipient: {notification.recipient}")
                return False
            
            # Create email message
            message = await self._create_message(notification)
            
            # Send email
            await self._send_email(
                notification.recipient,
                message
            )
            
            self.logger.info(f"Email sent to {notification.recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    async def validate_recipient(self, recipient: str) -> bool:
        """
        Validate email address format.
        
        Args:
            recipient: Email address to validate
            
        Returns:
            bool: True if valid email format
        """
        # Support multiple recipients separated by comma
        recipients = [r.strip() for r in recipient.split(',')]
        
        for email in recipients:
            if not self.email_pattern.match(email):
                return False
        
        return True
    
    async def _create_message(self, notification: Notification) -> MIMEMultipart:
        """
        Create email message from notification.
        
        Args:
            notification: Notification data
            
        Returns:
            MIMEMultipart: Email message
        """
        # Create message container
        msg = MIMEMultipart('alternative')
        
        # Set headers
        msg['Subject'] = notification.subject
        msg['From'] = f"{self.sender_name} <{self.sender_email}>"
        msg['To'] = notification.recipient
        
        # Add message ID for tracking
        msg['Message-ID'] = f"<{notification.id}@{self.smtp_host}>"
        
        # Add custom headers for tracking
        msg['X-Notification-ID'] = notification.id
        msg['X-Priority'] = str(notification.priority.value + 1)  # Email priority 1-5
        
        # Create plain text part
        text_part = MIMEText(notification.body, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Create HTML part if available
        html_body = notification.metadata.get('html_body')
        if html_body:
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
        else:
            # Convert plain text to simple HTML
            html_content = self._text_to_html(notification.body)
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
        
        # Add attachments if present
        attachments = notification.metadata.get('attachments', [])
        for attachment in attachments:
            await self._add_attachment(msg, attachment)
        
        return msg
    
    def _text_to_html(self, text: str) -> str:
        """
        Convert plain text to HTML.
        
        Args:
            text: Plain text content
            
        Returns:
            str: HTML content
        """
        # Escape HTML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Convert newlines to <br>
        text = text.replace('\n', '<br>\n')
        
        # Create HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="content">
                {text}
            </div>
            <div class="footer">
                <p>This is an automated message from AG News Classification System.</p>
                <p>Please do not reply to this email.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]) -> None:
        """
        Add attachment to email message.
        
        Args:
            msg: Email message
            attachment: Attachment data with 'path' or 'content' and 'filename'
        """
        try:
            # Get attachment data
            filename = attachment.get('filename', 'attachment')
            
            if 'path' in attachment:
                # Read from file
                path = Path(attachment['path'])
                with open(path, 'rb') as f:
                    content = f.read()
                filename = filename or path.name
            elif 'content' in attachment:
                # Use provided content
                content = attachment['content']
                if isinstance(content, str):
                    content = content.encode('utf-8')
            else:
                return
            
            # Create attachment
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(content)
            encoders.encode_base64(part)
            
            # Add header
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )
            
            # Attach to message
            msg.attach(part)
            
        except Exception as e:
            self.logger.error(f"Failed to add attachment: {e}")
    
    async def _send_email(self, recipient: str, message: MIMEMultipart) -> None:
        """
        Send email message via SMTP.
        
        Args:
            recipient: Recipient email address(es)
            message: Email message to send
        """
        # Parse multiple recipients
        recipients = [r.strip() for r in recipient.split(',')]
        
        # Send in batches if many recipients
        for i in range(0, len(recipients), self.max_recipients_per_message):
            batch = recipients[i:i + self.max_recipients_per_message]
            
            # Update To header for batch
            message.replace_header('To', ', '.join(batch))
            
            # Send batch
            await self._send_smtp(batch, message.as_string())
    
    async def _send_smtp(self, recipients: List[str], message_text: str) -> None:
        """
        Send message via SMTP connection.
        
        Args:
            recipients: List of recipient emails
            message_text: Message text to send
        """
        loop = asyncio.get_event_loop()
        
        async with self._connection_lock:
            try:
                # Create or reuse connection
                if self._smtp_connection is None:
                    await self._connect_smtp()
                
                # Send message
                await loop.run_in_executor(
                    None,
                    self._smtp_connection.sendmail,
                    self.sender_email,
                    recipients,
                    message_text
                )
                
            except (smtplib.SMTPException, ConnectionError) as e:
                # Reconnect and retry once
                self.logger.warning(f"SMTP error, reconnecting: {e}")
                await self._connect_smtp()
                
                await loop.run_in_executor(
                    None,
                    self._smtp_connection.sendmail,
                    self.sender_email,
                    recipients,
                    message_text
                )
    
    async def _connect_smtp(self) -> None:
        """Establish SMTP connection."""
        loop = asyncio.get_event_loop()
        
        def connect():
            # Create SMTP connection
            if self.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            # Authenticate if credentials provided
            if self.username and self.password:
                server.login(self.username, self.password)
            
            return server
        
        # Close existing connection
        if self._smtp_connection:
            try:
                self._smtp_connection.quit()
            except:
                pass
        
        # Create new connection
        self._smtp_connection = await loop.run_in_executor(None, connect)
        self.logger.info(f"Connected to SMTP server: {self.smtp_host}:{self.smtp_port}")
    
    async def send_batch(self, notifications: List[Notification]) -> Dict[str, bool]:
        """
        Send batch of email notifications.
        
        Args:
            notifications: List of notifications to send
            
        Returns:
            Dict[str, bool]: Results by notification ID
        """
        results = {}
        
        # Group by recipient for efficiency
        grouped = {}
        for notification in notifications:
            recipient = notification.recipient
            if recipient not in grouped:
                grouped[recipient] = []
            grouped[recipient].append(notification)
        
        # Send grouped notifications
        for recipient, group in grouped.items():
            # Combine notifications for same recipient
            if len(group) == 1:
                # Single notification
                results[group[0].id] = await self.send(group[0])
            else:
                # Multiple notifications - combine into digest
                digest = await self._create_digest(recipient, group)
                success = await self.send(digest)
                
                for notification in group:
                    results[notification.id] = success
        
        return results
    
    async def _create_digest(
        self,
        recipient: str,
        notifications: List[Notification]
    ) -> Notification:
        """
        Create digest notification from multiple notifications.
        
        Args:
            recipient: Recipient email
            notifications: List of notifications
            
        Returns:
            Notification: Digest notification
        """
        # Combine subjects
        subject = f"Notification Digest ({len(notifications)} items)"
        
        # Combine bodies
        body_parts = []
        for i, notif in enumerate(notifications, 1):
            body_parts.append(f"--- Notification {i} ---")
            body_parts.append(f"Subject: {notif.subject}")
            body_parts.append(notif.body)
            body_parts.append("")
        
        body = "\n".join(body_parts)
        
        # Create digest notification
        import uuid
        digest = Notification(
            id=str(uuid.uuid4()),
            channels=notifications[0].channels,
            recipient=recipient,
            subject=subject,
            body=body,
            priority=min(n.priority for n in notifications),
            metadata={"is_digest": True, "count": len(notifications)}
        )
        
        return digest
    
    def __del__(self):
        """Cleanup SMTP connection on deletion."""
        if hasattr(self, '_smtp_connection') and self._smtp_connection:
            try:
                self._smtp_connection.quit()
            except:
                pass
