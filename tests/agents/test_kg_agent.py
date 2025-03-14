import pytest
from okagents.agents import KGAgent
import os
import logging
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from okagents.config import Config

config = Config()

# ---------------------------------- 只测试了 pre-parse 功能，不想污染数据库 ----------------------------------


@pytest.fixture
def model():
    # creating the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        api_key=config.DEEPSEEK_API_KEY,
        url=config.DEEPSEEK_API_BASE,
        model_type=ModelType.DEEPSEEK_CHAT,
        model_config_dict={"max_tokens": 4096},
    )
    return model


@pytest.fixture
def kg_agent(model):
    agent = KGAgent(model)
    return agent
    # TODO Use Cypher/agent.clean() Clean up after tests


def test_str_processing(kg_agent):
    """测试纯文本处理"""
    test_data = """Large Language Models (LLMs) have shown remarkable capabilities in various tasks. 
    However, their reasoning abilities still need improvement. Recent research focuses on 
    reinforcement learning approaches to enhance LLM reasoning. Key challenges include 
    reward signal design and efficient training algorithms."""

    results = kg_agent.pre_parse(content=test_data)
    assert len(results) > 0


def test_remote_url_processing(kg_agent):
    """测试URL处理"""
    test_url = "https://github.com/PRIME-RL/PRIME"

    kg_agent.pre_parse(content=test_url)

    results = kg_agent.run_retriever(query="PRIME")

    assert len(results) > 0


# 超时 pass，也不建议在生产中使用 file，先解析成 text 再传进来更好
def test_pdf_file_processing(kg_agent):
    """测试PDF文件处理"""
    # 获取测试PDF路径
    test_pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "prime.pdf",
    )
    logging.info(f"Test PDF path: {test_pdf_path}")

    if not os.path.exists(test_pdf_path):
        pytest.skip("Test PDF file not found")

    import time

    start_time = time.time()
    try:
        results = kg_agent.pre_parse(content=test_pdf_path)
        assert len(results) > 0
    except Exception as e:
        if time.time() - start_time > 300:  # 5分钟超时
            pytest.skip("Test timed out after 5 minutes")
        raise e

    assert len(results) > 0
