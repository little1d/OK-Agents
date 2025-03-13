import pytest
from okagents.agents import MilvusAgent
import os
import logging


@pytest.fixture
def milvus_agent():
    agent = MilvusAgent(collection_name="test_collection")
    yield agent
    # Clean up after tests
    agent.clear()


def test_str_processing(milvus_agent):

    test_data = """Advanced reasoning of large language models (LLMs), while improvable through data-driven imitation, is still clouded by serious scalability challenges. We believe the key to overcoming such challenges lies in transforming data-driven approaches into exploration-based methods, as exemplified by reinforcement learning (RL). To this end, two critical bottlenecks need to be alleviated to bridge this transformation: (1) how to obtain precise reward signals efficiently and scalably, especially for dense ones? (2) how can we build effective RL algorithms to fully unleash the potential of these signals?
We seek the scalable path towards advanced reasoning capabilities with efficient reward modeling and reinforcement learning. Our work stems from the implicit process reward modeling (PRM) objective. Without the need for any process label, implicit PRM is trained as an outcome reward model (ORM) and then used as a PRM. Besides improving model performance through inference scaling, the true power of the implicit PRM is unveiled in online RL training. Specifically, it brings three benefits to RL:
Dense Reward: Implicit PRM directly learns a Q-function that provides rewards for each token, which alleviates the reward sparsity issue without the need of an extra value model.
Scalability: Implicit PRM can be online updated with only outcome label. Therefore, we can directly update the PRM with on-policy rollouts given outcome verifiers, which mitigates the distribution shift as well as scalability issues for PRMs.
Simplicity: Implicit PRM is inherently a language model. In practice, we show that it is unnecessary to train a PRM beforehand, since the SFT model itself already serves as a strong starting point.
"""

    milvus_agent.parse(content=str(test_data))

    results = milvus_agent.run(
        query="Dense Reward", top_k=1, similarity_threshold=0.1
    )

    assert len(results) > 0


def test_pdf_file_processing(milvus_agent):
    # Test PDF file processing
    test_pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "prime.pdf",
    )
    logging.info(f"Test PDF path: {test_pdf_path}")
    print(test_pdf_path)

    if not os.path.exists(test_pdf_path):
        pytest.skip("Test PDF file not found")

    milvus_agent.parse(content=test_pdf_path)

    results = milvus_agent.run(
        query="Dense Reward", top_k=1, similarity_threshold=0.1
    )

    assert len(results) > 0


def test_remote_url_processing(milvus_agent):
    test_url = "https://github.com/PRIME-RL/PRIME"
    milvus_agent.parse(content=test_url)
    results = milvus_agent.run(
        query="Dense Reward", top_k=1, similarity_threshold=0.1
    )
    assert len(results) > 0
