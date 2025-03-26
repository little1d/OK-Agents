#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-12 18:02:22
@File: src/agent/milvus_agent.py
@IDE: vscode
@Description:
    Milvus Agent
"""


from typing import Optional, Dict, Union, List
from camel.storages.vectordb_storages import MilvusStorage
from camel.embeddings import OpenAIEmbedding
from camel.retrievers import VectorRetriever
from camel.storages.vectordb_storages import VectorDBStatus
from okagents.config import Config
import logging
from okagents.config import Config

logger = logging.getLogger(__name__)
config = Config()


class MilvusAgent:
    def __init__(
        self,
        vector_dim: int = 1536,  # OpenAI embedding dimension
        collection_name: Optional[str] = None,
    ):
        """
        初始化 MilvusAgent

        Args:
            vector_dim (int): 向量维度，默认为 OpenAI 的 1536
            collection_name : str
        """
        # 初始化 Milvus 存储
        self.storage = MilvusStorage(
            vector_dim=vector_dim,
            url_and_api_key=(config.MILVUS_URL, ""),
            collection_name=collection_name,
        )

        # 初始化 VectorRetriever
        self.retriever = VectorRetriever(
            embedding_model=OpenAIEmbedding(
                url=config.OPENAI_API_BASE, api_key=config.OPENAI_API_KEY
            ),
            storage=self.storage,
        )

    def parse(
        self,
        content: Union[str, List[str]],
    ) -> None:
        """
        解析内容并存储到 Milvus，接受 URL, Str, Local file path
        """
        self.retriever.process(content=content)

    def run_retriever(
        self, query: str, top_k: int = 1, similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        执行检索，返回检索后的信息
        """
        # 执行检索
        retrieved_results = self.retriever.query(
            query=query, top_k=top_k, similarity_threshold=similarity_threshold
        )

        return retrieved_results

    def delete(self, ids: List[str]):
        self.storage.delete(ids=ids)

    def status(self) -> VectorDBStatus:
        return self.storage.status()

    def clear(self) -> None:
        self.storage.clear()
