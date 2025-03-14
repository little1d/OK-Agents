# OK-Agents: A Online Knowledge update Agent Built by [Camel](https://www.camel-ai.org/)

## Feature

- 高效内容解析与存储，支持 text, url 和 file
- 无需复杂 Cypher 查询与 neo4j 数据库进行交互
- 快速混合检索

## Introdunction

1. 把数据存储至 milvus 数据库和 neo4j 知识图谱中
2. 实现混合检索（待测试）
3. 循环增量存储（待实现）
4. 可以对已有的内容进行删除，更新（待实现）

## To Do

1. url 内容解析效果较差，可以增加预处理和后处理的逻辑

2. 支持更多 Cypher 原生功能

## Dependency Installation

```bash
pip install uv

uv pip install -e .
```
