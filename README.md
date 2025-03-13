# OK-Agent: A Online Knowledge update Agent Built by [Camel](https://www.camel-ai.org/)

## Feature

- 高效内容解析与存储
- 无需复杂 Cypher 查询与 neo4j 数据库进行交互
- 快速检索

## Introdunction

1. 把 json 数据存储至 milvus 数据库和 neo4j 知识图谱中（是否要定义 schema？）
2. 实现混合检索
3. 循环增量存储
4. 可以对已有的内容进行删除，更新

## Dependency Installation

```bash
pip install uv

uv pip install -e .
```
