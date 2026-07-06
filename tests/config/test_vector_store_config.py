from ragkit.config.vector_store_config import VectorStoreConfig


def test_create_vector_store_config():
    config = VectorStoreConfig()

    assert config.path == "./vector_db"
    assert config.collection_name == "ragkit"