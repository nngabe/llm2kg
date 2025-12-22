"""
Agent fine-tuning module for GraphRAG tool use and faithfulness.

This module provides:
- Tool definitions for graph_lookup, web_search, cypher_query, entity_resolve
- SFT data generation for multi-turn tool-use conversations
- DPO data generation for faithfulness training
- Trainers for agent SFT and DPO
"""

from .tools import AGENT_TOOLS, ToolSchema, get_tools_for_qwen3
from .data import ToolDataGenerator, ToolConversation, DPODataGenerator, DPOSample, AgentFormatter
from .training import AgentSFTTrainer, AgentDPOTrainer
