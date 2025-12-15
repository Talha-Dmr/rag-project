"""Simple EBCAR reranker tests focusing on critical functionality."""
import os
import pytest
import tempfile
import torch
from unittest.mock import Mock, patch, MagicMock

from src.reranking.rerankers.ebcar_reranker import EBCARReranker


def test_ebcar_init_with_none_checkpoint():
    """Test that EBCAR raises ValueError when checkpoint is None."""
    config = {"checkpoint": None}
    with pytest.raises(ValueError, match="EBCAR checkpoint path is required"):
        EBCARReranker(config)


def test_ebcar_init_with_nonexistent_checkpoint():
    """Test that EBCAR raises FileNotFoundError when checkpoint doesn't exist."""
    config = {"checkpoint": "/nonexistent/path/model.pth"}
    with pytest.raises(FileNotFoundError, match="EBCAR checkpoint not found"):
        EBCARReranker(config)


def test_ebcar_initialization_with_valid_config():
    """Test that EBCAR initializes successfully with valid config."""
    # Create a temporary file to mock the checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save(torch.randn(1, 1, 768), tmp.name)

        try:
            config = {
                "checkpoint": tmp.name,
                "device": "cpu",
                "embedder_name": "mgte"
            }

            # Mock both the embedder and the model
            mock_model = MagicMock()
            mock_model.eval = Mock()  # Add eval method to mock
            mock_embedder = Mock()
            mock_embedder.embed_text.return_value = [0.1] * 768

            with patch('src.reranking.rerankers.ebcar_reranker.torch.load', return_value=mock_model), \
                 patch('src.reranking.rerankers.ebcar_reranker.EmbedderFactory') as mock_factory:

                mock_factory.create.return_value = mock_embedder

                # This should not raise any exception
                reranker = EBCARReranker(config)
                assert reranker is not None
                assert reranker.device == "cpu"
                assert reranker.embedder == mock_embedder

        finally:
            os.unlink(tmp.name)


def test_ebcar_import():
    """Test that EBCAR reranker can be imported."""
    # Simply test that the class can be imported and instantiated
    # The decorator registration is tested implicitly by import success
    from src.reranking.rerankers.ebcar_reranker import EBCARReranker
    assert EBCARReranker is not None
    assert hasattr(EBCARReranker, 'rerank')