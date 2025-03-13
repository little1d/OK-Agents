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
from camel.retrievers import AutoRetriever
from camel.messages import BaseMessage


class KGAgent:
    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        model: Optional[BaseModelBackend] = None,
        database: str = "neo4j",
        timeout: Optional[float] = None,
        truncate: bool = False,
        max_nodes: int = 100,
        max_relationships: int = 200,
    ):
        """
        初始化 KGAgent，集成 CAMEL 和 模型

        Args:
            neo4j_url (str): Neo4j 数据库 URL
            neo4j_username (str): Neo4j 用户名
            neo4j_password (str): Neo4j 密码
            model (Optional[BaseModelBackend]): 模型后端，默认为 Mistral Large 2
            database (str): 数据库名称，默认为 "neo4j"
            timeout (Optional[float]): 超时时间
            truncate (bool): 是否截断大列表
            max_nodes (int): 最大节点数限制
            max_relationships (int): 最大关系数限制
        """
        self.uio = UnstructuredIO()
        # 初始化 Knowledge Graph Agent
        self.kg_agent = KnowledgeGraphAgent(model=model)

        # 初始化 Neo4j 连接
        self.neo4j_graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=database,
            timeout=timeout,
            truncate=truncate,
        )

        # 配置参数
        self.max_nodes = max_nodes
        self.max_relationships = max_relationships

    def parse(self, content: str, id: int) -> str:
        """
        解析非结构化内容为知识图谱相关内容，目前只支持 text，返回 str
        """
        # 使用 model 进行内容解析
        element = self.uio.create_element_from_text(
            text=content, element_id=id
        )
        # ---------------------------------- 支持 file or url ----------------------------------

        # elements = self.uio.parse_file_or_url(input_path="")
        # chunk_elements = self.uio.chunk_elements(
        #     chunk_type="chunk_by_title", elements=elements
        # )
        # graph_elements = []
        # for chunk in chunk_elements:
        #     graph_element = self.kg_agent.run(chunk, parse_graph_elements=True)
        #     graph_elements.append(graph_element)
        # Let Knowledge Graph Agent extract node and relationship information
        parsed = self.kg_agent.run(element=element, parse_graph_elements=False)

        return parsed

    def validate(self, str) -> str:
        """
        验证知识图谱元素，额外用一个 ChatAgent
        """
        pass

    def save(self, content: str, id: int) -> None:
        """
        将知识图谱元素保存到 Neo4j
        """
        graph_elements = self.parse(content=content, id=id)
        self.neo4j_graph.add_graph_elements(graph_elements=[graph_elements])

    def run(
        self,
        query: List[str],
    ) -> Any:
        """
        运行检索和推理

        Args:
            query: 查询语句

        Returns:
            检索内容
        """
        query_element = self.uio.create_element_from_text(
            text=query,
        )
        # Let Knowledge Graph Agent extract node and relationship information from the qyery
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

        # Show the result from knowledge graph database
        print(kg_result)
        return kg_result

    def update(self, content: Union[str, "Element"], **kwargs) -> None:
        """更新Knowledge Graph"""
        pass
