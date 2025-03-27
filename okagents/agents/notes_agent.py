#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-26 14:24:02
@File: okagents/agents/chat_agent.py
@IDE: vscode
@Description:
     Notes Agent for extracting and managing key notes from reasoning data/preprocess data
"""

from typing import Optional, List, Dict, Any, Type, Union
from pydantic import BaseModel
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from okagents.agents.kg_agent import KGAgent
from okagents.agents.milvus_agent import MilvusAgent
import json
from camel.messages import BaseMessage


class BaseNotesResponse(BaseModel):
    """Base response format for notes extraction. All integrated classes must return str or List[str]"""

    notes: List[str]


class NotesAgent:
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        kg_agent: Optional[KGAgent] = None,
        milvus_agent: Optional[MilvusAgent] = None,
        system_message: str = "You are an expert at extracting key notes from scientific data or complex reasoning data..",
        tools: Optional[List[Any]] = None,
    ):
        """
        Initialize Notes Agent for processing reasoning data

        Args:
            model: Backend model for processing
            kg_agent: Pre-initialized KGAgent instance
            milvus_agent: Pre-initialized MilvusAgent instance
            system_message: System message for the chat agent
            tools: List of tools for the agent to use
        """
        self.chat_agent = ChatAgent(
            model=model, system_message=system_message, tools=tools or []
        )
        self.kg_agent = kg_agent or KGAgent(model=model)
        self.milvus_agent = milvus_agent or MilvusAgent()

        self._init_prompt_templates()

    def _init_prompt_templates(self):
        """Initialize default prompt templates"""
        self.DEFAULT_EXTRACTION_PROMPT = """
        Extract the most important factual notes from this reasoning data.
        Focus on concrete facts, entities, relationships and key findings.
        Return only the most essential information in bullet points.
        Reasoning data: {input}
        """

        self.DEFAULT_EXPERIMENT_PROMPT = """
        Extract and structure key information from this experimental setup.
        Include: reactants, solvents, conditions, concentrations, and other critical parameters.
        Format as clear bullet points with precise values where available.
        Experimental setup: {input}
        """

    def extract_notes(
        self,
        reasoning_data: str,
        save_schema: Type[BaseNotesResponse] = BaseNotesResponse,
        prompt: Optional[str] = None,
    ) -> BaseNotesResponse:
        """
        Extract key notes from reasoning data and store to storage systems

        Args:
            reasoning_data: Raw reasoning output to process
            response_model: Custom response model class
            prompt: Custom extraction prompt (uses default if None)

        Returns:
            Structured response with extracted notes
        """
        final_prompt = (prompt or self.DEFAULT_EXTRACTION_PROMPT).format(
            input=reasoning_data
        )

        response = self.chat_agent.step(
            final_prompt, response_format=save_schema
        )

        self._store_notes(response)
        return response

    def extract_experiment_info(
        self,
        experiment_data: str,
        save_schema: Type[BaseNotesResponse] = BaseNotesResponse,
        prompt: Optional[str] = None,
        enable_research: bool = False,
    ) -> BaseNotesResponse:
        """
        Extract and store experimental setup information

        Args:
            experiment_data: Description of experimental setup
            save_schema: Response schema class
            prompt: Custom extraction prompt (uses default if None)
            enable_research: Whether to enable web research for missing info

        Returns:
            Structured response with extracted notes
        """
        final_prompt = (prompt or self.DEFAULT_EXPERIMENT_PROMPT).format(
            input=experiment_data
        )

        # TODO 增加 web search 的 tool
        if enable_research:
            final_prompt += (
                "\n\nResearch and supplement any missing critical information."
            )

        response = self.chat_agent.step(
            final_prompt,
            response_format=save_schema,
        )
        print(f"response content: {response}")
        print(f"\nresponse type: {type(response)}")
        self._store_notes(response)
        return response

    def _store_notes(self, response: Union[BaseNotesResponse, List[str]]):
        """Internal method to store notes to both knowledge systems

        Args:
            response: 可以是以下类型之一：
                - camel.ChatAgentResponse: ChatAgentResponse.msg 是 camel.BaseMessage (包含JSON格式内容)
                - 字符串列表，所有字段必须是str或List[str]
        """
        if isinstance(response.msg, BaseMessage):
            try:
                # 提取JSON内容部分 (去除可能的代码块标记)
                content = (
                    response.msg.content.replace('```json\n', '')
                    .replace('```', '')
                    .strip()
                )
                data = json.loads(content)
                # 收集所有可能的文本字段
                text_parts = "\n".join(
                    str(item)
                    for value in data.values()
                    for item in (value if isinstance(value, list) else [value])
                )
            except (json.JSONDecodeError, AttributeError) as e:
                raise ValueError(
                    f"Failed to parse message content: {e}"
                ) from e

        # 处理纯列表输入
        elif isinstance(response, (list, tuple)):
            text_parts = "\n".join(str(item) for item in response)
        else:
            raise TypeError(
                f"Unsupported type, response attributes must be str or List[str]"
            )

        if text_parts:
            self.kg_agent.parse(text_parts)
            self.milvus_agent.parse(text_parts)

    def query_notes(
        self, query: str, top_k: int = 3, similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query relevant notes from storage systems, including KG and vector DB

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            Combined results from KG and vector DB
        """
        kg_results = self.kg_agent.run_retriever(query)
        milvus_results = self.milvus_agent.run_retriever(
            query, top_k=top_k, similarity_threshold=similarity_threshold
        )

        return {"knowledge_graph": kg_results, "vector_db": milvus_results}

    def structured_query(
        self, prompt: str, response_schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Execute a structured query with custom response format

        Args:
            prompt: User query
            response_model: Custom pydantic model for response

        Returns:
            Structured response matching the provided model
        """
        return self.chat_agent.step(prompt, response_format=response_schema)
