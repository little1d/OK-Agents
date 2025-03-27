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
        self.camel_kg_agent = KnowledgeGraphAgent(model=model)

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
    ) -> str:
        """
        References:
            https://github.com/camel-ai/camel/blob/master/camel/retrievers/vector_retriever.py

        解析非结构化内容为知识图谱相关内容，支持 file、url 或纯文本。
        自动判断内容类型并执行分块处理。
        """
        if isinstance(content, Element):
            # 如果已经是 Element 类型，直接处理
            return self._process_single_element(content)

        elif isinstance(content, str):
            # 检查是否是 URL
            parsed_url = urlparse(content)
            is_url = all([parsed_url.scheme, parsed_url.netloc])

            if is_url or os.path.exists(content):
                # URL 或文件路径的处理
                return self._process_url_or_file(content, is_url)
            else:
                # 纯文本的处理
                return self._process_plain_text(content)

        return ""

    def _process_single_element(self, element: Element) -> str:
        """处理单个 Element"""
        return self.camel_kg_agent.run(
            element=element,
            parse_graph_elements=False,
        )

    def _process_plain_text(self, text: str) -> str:
        """处理纯文本"""
        print(f"Parsed content as plain text: {text}")
        element = self.uio.create_element_from_text(text=text)
        return self._process_single_element(element)

    def _process_url_or_file(self, path: str, is_url: bool) -> str:
        """处理 URL 或文件"""
        print(f"Parsed content from {'URL' if is_url else 'file'}: {path}")
        element_lists = self.uio.parse_file_or_url(input_path=path) or []
        if not element_lists:
            warnings.warn(f"No elements were extracted from: {path}")
            return ""

        chunk_elements = self.uio.chunk_elements(
            chunk_type="chunk_by_title", elements=element_lists
        )

        results = []
        for chunk in chunk_elements:
            parsed = self.camel_kg_agent.run(
                element=chunk,
                parse_graph_elements=False,
            )
            results.append(parsed)

        return "\n".join(results)

    def validate(self, content: str) -> str:
        """
        验证pre-parsed 的准确性，并做出验证和删减（避免内容过多重复），额外用一个 ChatAgent/其他策略
        """
        # TODO 添加验证的逻辑  获取当前知识库节点和关系 --> 进行内容删减
        return content

    def parse(
        self,
        content: str,
    ) -> None:
        """
        解析文件，并转换为 node 和 relation，再将知识图谱元素保存到 Neo4j

        Args:
            content (str): 要解析的内容
            should_chunk (bool): 是否进行分块处理，默认为True
            max_characters (int): 分块的最大字符数，默认为500
        """
        # 预处理内容
        pre_parsed = self.pre_parse(content=content)
        # validate
        validated_content = self.validate(content=pre_parsed)
        # parse str to element
        elements = self.uio.create_element_from_text(validated_content)
        graph_element = self.camel_kg_agent.run(
            element=elements,
            parse_graph_elements=True,
        )
        # save
        self.neo4j_graph.add_graph_elements(
            graph_elements=[graph_element],
        )
        print(f"Save successfully, content: {graph_element}")
        return graph_element

    def run_retriever(
        self,
        query: str,
    ) -> Any:
        """
        运行检索和推理
        """
        query_element = self.uio.create_element_from_text(
            text=query,
        )
        # Let Knowledge Graph Agent extract node and relationship information from the query
        ans_element = self.camel_kg_agent.run(
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
            result = self.neo4j_graph.query(query=n4j_query)
            kg_result.extend(result)

        kg_result = [item['Description'] for item in kg_result]

        return kg_result

    def update(self, content: Union[str, "Element"], **kwargs) -> None:
        """更新Knowledge Graph"""
        pass
