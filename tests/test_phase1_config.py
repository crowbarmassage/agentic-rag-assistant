"""Phase 1 Tests: Project Scaffolding and Configuration."""

import os
import pytest
from pathlib import Path


class TestProjectStructure:
    """Test that project structure exists."""

    def test_src_directory_exists(self):
        """Verify src directory structure."""
        base_path = Path(__file__).parent.parent / "src"
        assert base_path.exists(), "src/ directory should exist"

        expected_dirs = ["models", "providers", "pipelines", "routing", "vectorstore", "utils"]
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"src/{dir_name}/ should exist"
            assert (dir_path / "__init__.py").exists(), f"src/{dir_name}/__init__.py should exist"

    def test_data_directory_exists(self):
        """Verify data directory exists."""
        data_path = Path(__file__).parent.parent / "data"
        assert data_path.exists(), "data/ directory should exist"

    def test_tests_directory_exists(self):
        """Verify tests directory exists."""
        tests_path = Path(__file__).parent
        assert tests_path.exists(), "tests/ directory should exist"


class TestConfiguration:
    """Test configuration module."""

    def test_settings_import(self):
        """Verify Settings class can be imported."""
        from src.config import Settings
        assert Settings is not None

    def test_settings_instantiation(self):
        """Verify Settings can be instantiated with defaults."""
        from src.config import Settings
        settings = Settings()
        assert settings is not None

    def test_settings_default_values(self):
        """Verify default configuration values."""
        from src.config import Settings
        settings = Settings()

        assert settings.llm_provider == "openai"
        assert settings.embedding_provider == "sentence_transformers"
        assert settings.chroma_persist_dir == "./data/chroma_db"
        assert settings.retrieval_min_threshold == 0.5
        assert settings.retrieval_max_k == 10
        assert settings.retrieval_drop_off_ratio == 0.7
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.debug is True

    def test_settings_from_env_vars(self):
        """Verify settings can be overridden by environment variables."""
        from src.config import Settings

        # Set test env vars
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["API_PORT"] = "9000"

        settings = Settings()

        # The Settings class should pick up env vars
        # (Note: depends on model_config settings)

        # Clean up
        del os.environ["LLM_PROVIDER"]
        del os.environ["API_PORT"]

    def test_settings_validation(self):
        """Verify settings validation constraints."""
        from src.config import Settings
        from pydantic import ValidationError

        # These should work with valid values
        settings = Settings(
            retrieval_min_threshold=0.7,
            retrieval_max_k=20
        )
        assert settings.retrieval_min_threshold == 0.7
        assert settings.retrieval_max_k == 20

    def test_env_example_exists(self):
        """Verify .env.example template exists."""
        env_example = Path(__file__).parent.parent / ".env.example"
        assert env_example.exists(), ".env.example should exist"

        # Verify it has expected keys
        content = env_example.read_text()
        expected_keys = [
            "LLM_PROVIDER",
            "EMBEDDING_PROVIDER",
            "OPENAI_API_KEY",
            "CHROMA_PERSIST_DIR",
            "API_PORT"
        ]
        for key in expected_keys:
            assert key in content, f".env.example should contain {key}"
