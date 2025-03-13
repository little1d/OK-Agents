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

from okagents.config import Config

config = Config()


class KGAgent:
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        truncate: bool = False,
        max_nodes: int = 100,
        max_relationships: int = 200,
    ):
        """
        初始化 KGAgent，集成 CAMEL 和 模型

        Args:
            model (Optional[BaseModelBackend]): 模型后端
            truncate (bool): 是否截断大列表
            max_nodes (int): 最大节点数限制
            max_relationships (int): 最大关系数限制
        """
        self.uio = UnstructuredIO()
        # 初始化 Knowledge Graph Agent
        self.kg_agent = KnowledgeGraphAgent(model=model)

        # 初始化 Neo4j 连接
        self.neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URL,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            truncate=truncate,
        )

        # 配置参数
        self.max_nodes = max_nodes
        self.max_relationships = max_relationships

    def pre_parse(self, content: str, id: int, prompt: Optional[str]) -> str:
        """
        解析非结构化内容为知识图谱相关内容，目前只支持 text，返回 str
        # TODO 目前 prompt 使用 camel 自定义的 system propmpt，返回的是 GraphElement 类似的 str形式。
        # 因为增加了 validate 部分，所以其实并不用严格返回，可以留到后面处理，需要自定义 prompt！！！
        # TODO  https://docs.camel-ai.org/camel.loaders.html#camel.loaders.unstructured_io.UnstructuredIO
        # 自动支持 url, file, json
        """
        # ---------------------------------- 支持 text ----------------------------------
        graph_element = self.uio.create_element_from_text(
            text=content, element_id=id
        )

        # ---------------------------------- parse_file_or_url ----------------------------------

        # elements = self.uio.parse_file_or_url(input_path="")
        # chunk_elements = self.uio.chunk_elements(
        #     chunk_type="chunk_by_title", elements=elements
        # )
        # graph_elements = []
        # for chunk in chunk_elements:
        #     graph_element = self.kg_agent.run(chunk, parse_graph_elements=True)
        #     graph_elements.append(graph_element)
        # Let Knowledge Graph Agent extract node and relationship information

        # run the agent to extract node and relationship information
        # TODO 添加 prompt 参数，限制其不返回 GraphElement 形式，并限制字数...
        pre_parsed_elements = self.kg_agent.run(
            element=graph_element,
            parse_graph_elements=False,
        )

        return pre_parsed_elements

    def validate(self, content: str) -> str:
        """
        验证pre-parsed 的准确性，并做出验证和删减（避免内容过多重复），额外用一个 ChatAgent/其他策略
        """
        # TODO 添加验证的逻辑  获取当前知识库节点和关系 --> 进行内容删减
        return content

    def parse(self, content: str, id: int) -> None:
        """
        解析文件，并转换为 node 和 relation，再将知识图谱元素保存到 Neo4j
        """
        pre_parsed = self.pre_parse(content=content, id=id)
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
