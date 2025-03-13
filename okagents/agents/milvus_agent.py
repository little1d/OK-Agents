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
from camel.types import StorageType
from camel.retrievers import AutoRetriever
from camel.storages.vectordb_storages import VectorRecord, VectorDBStatus
from okagents.config import Config
import logging
import datetime

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

        # 初始化 AutoRetriever
        self.retriever = AutoRetriever(
            url_and_api_key=(
                config.MILVUS_URL,  # Your Milvus connection URL
                "",  # Your Milvus token, default None
            ),
            storage_type=StorageType.MILVUS,
            embedding_model=OpenAIEmbedding(
                url=config.OPENAI_API_BASE, api_key=config.OPENAI_API_KEY
            ),
        )

    def parse(
        self,
        content: Union[str, List[str]],
        batch_size: int = 100,
    ) -> None:
        """
        解析内容并存储到 Milvus

        Args:
            content: 要存储的内容，可以是字符串或字符串列表
            batch_size: 批量处理大小
        """
        if isinstance(content, str):
            content = [content]

        # 批量处理内容
        for i in range(0, len(content), batch_size):
            batch = content[i : i + batch_size]

            # 生成向量记录
            records = []
            for idx, text in enumerate(batch):
                record = VectorRecord(
                    id=f"doc_{i + idx}",
                    vector=self.retriever.embed(text),
                    payload={
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                records.append(record)

            # 存储到 Milvus
            self.storage.add(records)

    def run(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        执行检索，返回检索后的信息
        """
        # 执行检索
        retrived_results = self.retriever.run_vector_retriever(
            query=query, top_k=top_k, return_detailed_info=True
        )

        return retrived_results

    def delete(self, ids: List[str]):
        self.storage.delete(ids=ids)

    def status(self) -> VectorDBStatus:
        self.storage.status

    def clear(self) -> None:
        self.storage.clear
