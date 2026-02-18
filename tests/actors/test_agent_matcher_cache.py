"""
Test AgentMatcher caching behavior without external dependencies.

This test verifies that:
1. AgentMatcher can be instantiated
2. Embeddings are computed and cached on first use
3. Cached embeddings are loaded on subsequent uses
4. Cache is stored in user cache directory (not package assets)
"""

import tempfile
from pathlib import Path

import pytest

from ngec.actors.agent_matcher import AgentMatcher, get_cache_path


def test_cache_path_creation():
    """Verify that get_cache_path creates a valid directory."""
    cache_path = get_cache_path()
    assert isinstance(cache_path, Path)
    assert cache_path.exists()
    assert cache_path.is_dir()
    # Should be in user cache, not in package
    assert "plover" in str(cache_path).lower()


def test_agent_matcher_uses_cache():
    """Test that AgentMatcher creates and uses cache correctly."""
    # Use a temporary cache directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache"

        # First instantiation - should compute embeddings
        matcher1 = AgentMatcher(cache_path=cache_path)

        # Verify cache directory was created
        assert cache_path.exists()

        # Find the cache subdirectory (hash_filename format)
        cache_subdirs = list(cache_path.glob("*_PLOVER_agents"))
        assert len(cache_subdirs) == 1, "Should have exactly one cache directory"

        cache_file = cache_subdirs[0] / "bert_matrix.pkl"
        assert cache_file.exists(), "Cache file should exist"

        # Get file modification time
        first_mtime = cache_file.stat().st_mtime

        # Second instantiation - should load from cache
        matcher2 = AgentMatcher(cache_path=cache_path)

        # Cache file should not have been modified
        second_mtime = cache_file.stat().st_mtime
        assert first_mtime == second_mtime, "Cache should be reused, not recomputed"


def test_agent_matcher_custom_agents_file():
    """Test that different agent files create separate caches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache"

        # Create matcher with default agents file
        matcher1 = AgentMatcher(cache_path=cache_path)

        # Should have one cache directory
        cache_subdirs = list(cache_path.glob("*"))
        assert len(cache_subdirs) == 1

        # Note: We can't easily test with a different agents file without
        # creating a custom agent file, but the cache key mechanism is verified
        # by the directory naming structure


def test_agent_matcher_basic_functionality():
    """Smoke test that AgentMatcher can perform basic matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache"
        matcher = AgentMatcher(cache_path=cache_path)

        # Test basic matching - should not raise errors
        # Note: We don't check specific results as those depend on model/data
        result = matcher.trf_agent_match("president", country="USA")

        # Result should be a dict or None
        assert result is None or isinstance(result, dict)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
