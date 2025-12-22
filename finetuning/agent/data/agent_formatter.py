"""
Agent conversation formatter for Qwen3 tool-calling format.

Formats multi-turn tool-use conversations into the proper format
for SFT training with Qwen3 models.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..tools import get_tools_for_qwen3, AGENT_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Qwen3 special tokens for tool calling
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"


@dataclass
class FormattedExample:
    """A formatted training example."""
    text: str  # Full formatted conversation text
    messages: List[Dict[str, Any]]  # Structured messages
    num_turns: int
    has_tool_calls: bool


class AgentFormatter:
    """
    Format tool-use conversations for Qwen3 training.

    Supports multiple formats:
    - qwen3: Native Qwen3 tool-calling format with special tokens
    - chatml: ChatML format with function calling (OpenAI style)
    """

    def __init__(
        self,
        format_type: str = "qwen3",
        include_system_prompt: bool = True,
        max_tool_response_length: int = 2000,
    ):
        self.format_type = format_type
        self.include_system_prompt = include_system_prompt
        self.max_tool_response_length = max_tool_response_length
        self.tools = get_tools_for_qwen3()

    def get_system_prompt(self) -> str:
        """Get the system prompt for tool-use conversations."""
        tool_descriptions = []
        for tool in AGENT_TOOLS:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        tools_str = "\n".join(tool_descriptions)

        return f"""You are a helpful assistant with access to the following tools:

{tools_str}

When you need to use a tool, respond with a tool call in the following format:
{TOOL_CALL_START}
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
{TOOL_CALL_END}

After receiving tool results, synthesize the information to provide a helpful, accurate response.
Always base your answers on the information retrieved from tools. If you cannot find the information,
say so rather than making up an answer."""

    def format_conversation(self, conversation: Dict[str, Any]) -> Optional[FormattedExample]:
        """
        Format a single conversation for training.

        Args:
            conversation: Dict with 'turns' key containing list of turn dicts

        Returns:
            FormattedExample or None if formatting fails
        """
        turns = conversation.get("turns", [])
        if not turns:
            return None

        if self.format_type == "qwen3":
            return self._format_qwen3(turns)
        elif self.format_type == "chatml":
            return self._format_chatml(turns)
        else:
            logger.warning(f"Unknown format type: {self.format_type}")
            return self._format_qwen3(turns)

    def _format_qwen3(self, turns: List[Dict[str, Any]]) -> FormattedExample:
        """Format for Qwen3 native tool-calling."""
        messages = []
        text_parts = []
        has_tool_calls = False

        # Add system prompt
        if self.include_system_prompt:
            system_msg = {"role": "system", "content": self.get_system_prompt()}
            messages.append(system_msg)
            text_parts.append(f"<|im_start|>system\n{system_msg['content']}<|im_end|>")

        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            tool_calls = turn.get("tool_calls", [])
            tool_call_id = turn.get("tool_call_id", "")

            if role == "user":
                msg = {"role": "user", "content": content}
                messages.append(msg)
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")

            elif role == "assistant":
                if tool_calls:
                    # Assistant makes tool call(s)
                    has_tool_calls = True
                    tool_call_strs = []
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        name = func.get("name", "")
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                pass
                        tool_call_strs.append(
                            f'{TOOL_CALL_START}\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n{TOOL_CALL_END}'
                        )

                    tool_content = "\n".join(tool_call_strs)
                    msg = {"role": "assistant", "content": tool_content, "tool_calls": tool_calls}
                    messages.append(msg)
                    text_parts.append(f"<|im_start|>assistant\n{tool_content}<|im_end|>")

                elif content:
                    # Regular assistant response
                    msg = {"role": "assistant", "content": content}
                    messages.append(msg)
                    text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

            elif role == "tool":
                # Tool response
                has_tool_calls = True
                tool_content = content
                if isinstance(tool_content, dict):
                    tool_content = json.dumps(tool_content)

                # Truncate long tool responses
                if len(tool_content) > self.max_tool_response_length:
                    tool_content = tool_content[:self.max_tool_response_length] + "... [truncated]"

                formatted_response = f"{TOOL_RESPONSE_START}\n{tool_content}\n{TOOL_RESPONSE_END}"
                msg = {"role": "tool", "content": formatted_response, "tool_call_id": tool_call_id}
                messages.append(msg)
                text_parts.append(f"<|im_start|>tool\n{formatted_response}<|im_end|>")

        # Join all parts
        full_text = "\n".join(text_parts)

        return FormattedExample(
            text=full_text,
            messages=messages,
            num_turns=len(turns),
            has_tool_calls=has_tool_calls,
        )

    def _format_chatml(self, turns: List[Dict[str, Any]]) -> FormattedExample:
        """Format for ChatML/OpenAI style."""
        messages = []
        text_parts = []
        has_tool_calls = False

        # Add system prompt
        if self.include_system_prompt:
            system_msg = {"role": "system", "content": self.get_system_prompt()}
            messages.append(system_msg)
            text_parts.append(f"<|im_start|>system\n{system_msg['content']}<|im_end|>")

        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            tool_calls = turn.get("tool_calls", [])
            tool_call_id = turn.get("tool_call_id", "")

            if role == "user":
                msg = {"role": "user", "content": content}
                messages.append(msg)
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")

            elif role == "assistant":
                if tool_calls:
                    has_tool_calls = True
                    # OpenAI style: function_call in message
                    msg = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls,
                    }
                    messages.append(msg)

                    # Format for text
                    tc_str = json.dumps(tool_calls, indent=2)
                    text_parts.append(f"<|im_start|>assistant\n[TOOL_CALLS]\n{tc_str}<|im_end|>")
                else:
                    msg = {"role": "assistant", "content": content}
                    messages.append(msg)
                    text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

            elif role == "tool":
                has_tool_calls = True
                tool_content = content
                if isinstance(tool_content, dict):
                    tool_content = json.dumps(tool_content)

                if len(tool_content) > self.max_tool_response_length:
                    tool_content = tool_content[:self.max_tool_response_length] + "... [truncated]"

                msg = {
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call_id,
                }
                messages.append(msg)
                text_parts.append(f"<|im_start|>tool ({tool_call_id})\n{tool_content}<|im_end|>")

        full_text = "\n".join(text_parts)

        return FormattedExample(
            text=full_text,
            messages=messages,
            num_turns=len(turns),
            has_tool_calls=has_tool_calls,
        )

    def format_dataset(
        self,
        conversations: List[Dict[str, Any]],
        min_turns: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Format a list of conversations for training.

        Args:
            conversations: List of conversation dicts
            min_turns: Minimum turns required (filter out shorter)

        Returns:
            List of dicts with 'text' key for training
        """
        formatted = []
        for conv in conversations:
            example = self.format_conversation(conv)
            if example and example.num_turns >= min_turns:
                formatted.append({
                    "text": example.text,
                    "num_turns": example.num_turns,
                    "has_tool_calls": example.has_tool_calls,
                })

        logger.info(f"Formatted {len(formatted)}/{len(conversations)} conversations")

        # Log statistics
        with_tools = sum(1 for f in formatted if f["has_tool_calls"])
        logger.info(f"  {with_tools} conversations have tool calls")

        return formatted

    def format_for_tokenizer(
        self,
        conversation: Dict[str, Any],
        tokenizer,
        max_length: int = 8192,
    ) -> Optional[Dict[str, Any]]:
        """
        Format and tokenize a conversation.

        Args:
            conversation: Conversation dict
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        example = self.format_conversation(conversation)
        if not example:
            return None

        # Use tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                encoded = tokenizer.apply_chat_template(
                    example.messages,
                    tokenize=True,
                    max_length=max_length,
                    truncation=True,
                    return_dict=True,
                    tools=self.tools,
                )
                return encoded
            except Exception as e:
                logger.debug(f"Chat template failed: {e}, using text format")

        # Fallback to direct text encoding
        encoded = tokenizer(
            example.text,
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()

        return encoded
