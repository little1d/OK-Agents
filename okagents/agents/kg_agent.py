#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-13 11:19:39
@File: src/agent/kg_agent.py
@IDE: vscode
@Description:
    Knowledge Graph Agent using Neo4j
"""
from camel.storages import Neo4jGraph
from camel.storages.graph_storages import GraphElement
from camel.agents import KnowledgeGraphAgent, ChatAgent
from camel.models import BaseModelBackend
from unstructured.documents.elements import Element
from typing import Optional, Any, Union, List
from camel.loaders import UnstructuredIO
import os
import warnings
from urllib.parse import urlparse

from okagents.config import Config

config = Config()


class KGAgent:
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
    ):
        """
        初始化 KGAgent，集成 CAMEL 和 模型

        Args:
            model (Optional[BaseModelBackend]): 模型后端
        """
        self.uio = UnstructuredIO()
        # 初始化 Knowledge Graph Agent
        self.kg_agent = KnowledgeGraphAgent(model=model)

        # 初始化 Neo4j 连接
        self.neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URL,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
        )

    def pre_parse(
        self,
        content: Union[str, 'Element'],
        prompt: Optional[str] = None,
        should_chunk: bool = True,
        max_characters=500,
    ) -> str:
        """
        References:
            https://github.com/camel-ai/camel/blob/master/camel/retrievers/vector_retriever.py

        解析非结构化内容为知识图谱相关内容，支持 file、url 或纯文本。
        自动判断内容类型并执行分块处理。

        # TODO 目前 prompt 使用 camel 自定义的 system propmpt，返回的是 GraphElement 类似的 str形式。
        # 因为增加了 validate 部分，所以其实并不用严格返回，可以留到后面处理，需要自定义 prompt！！！
        """
        elements = []

        if isinstance(content, Element):
            elements = [content]

        elif isinstance(content, str):
            # 检查是否是 URL
            parsed_url = urlparse(content)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if is_url or os.path.exists(content):
                # 如果是 URL 或文件路径，解析文件或 URL
                elements = self.uio.parse_file_or_url(input_path=content) or []
                print(
                    f"Parsed content from {'URL' if is_url else 'file'}: {content}"
                )
            else:
                # 如果是纯文本，创建文本元素
                elements = [
                    self.uio.create_element_from_text(
                        text=content,
                    )
                ]
                print(f"Parsed content as plain text : {elements}")

        if not elements:
            warnings.warn(
                f"No elements were extracted from the content: {content}"
            )
        else:
            # Chunk the content if required
            chunks = (
                self.uio.chunk_elements(
                    chunk_type="chunk_by_title",
                    elements=elements,
                    max_characters=max_characters,
                )
                if should_chunk
                else elements
            )
            print(f"Chunked content into {len(chunks)} parts")

            # 运行知识图谱代理提取信息
            pre_parsed_elements = self.kg_agent.run(
                element=chunks,
                parse_graph_elements=False,
            )

        return pre_parsed_elements

    def validate(self, content: str) -> str:
        """
        验证pre-parsed 的准确性，并做出验证和删减（避免内容过多重复），额外用一个 ChatAgent/其他策略
        """
        # TODO 添加验证的逻辑  获取当前知识库节点和关系 --> 进行内容删减
        return content

    def parse(self, content: str) -> None:
        """
        解析文件，并转换为 node 和 relation，再将知识图谱元素保存到 Neo4j
        """
        pre_parsed = self.pre_parse(content=content)
        validate = self.validate(pre_parsed)
        # 转换为 GraphElement 对象，以添加进 Knowledge Graph
        graph_elements = self.kg_agent.run(
            element=validate, parse_graph_elements=True
        )
        self.neo4j_graph.add_graph_elements(
            graph_elements=[graph_elements],
        )

    def run(
        self,
        query: List[str],
    ) -> Any:
        """
        运行检索和推理
        """
        query_element = self.uio.create_element_from_text(
            text=query,
        )
        # Let Knowledge Graph Agent extract node and relationship information from the query
        ans_element = self.kg_agent.run(
            query_element, parse_graph_elements=True
        )
        # Match the enetity got from query in the knowledge graph storage content
        kg_result = []
        for node in ans_element.nodes:
            n4j_query = f"""
        MATCH (n {{id: '{node.id}'}})-[r]->(m)
        RETURN 'Node ' + n.id + ' (label: ' + labels(n)[0] + ') has relationship ' + type(r) + ' with Node ' + m.id + ' (label: ' + labels(m)[0] + ')' AS Description
        UNION
        MATCH (n)<-[r]-(m {{id: '{node.id}'}})
        RETURN 'Node ' + m.id + ' (label: ' + labels(m)[0] + ') has relationship ' + type(r) + ' with Node ' + n.id + ' (label: ' + labels(n)[0] + ')' AS Description
        """
            result = self.n4j.query(query=n4j_query)
            kg_result.extend(result)

        kg_result = [item['Description'] for item in kg_result]

        return kg_result

    def update(self, content: Union[str, "Element"], **kwargs) -> None:
        """更新Knowledge Graph"""
        pass
