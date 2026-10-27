import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import coordinator
from coordinator import expand_query_hyde, SearchQuery, distributed_search


@pytest.mark.asyncio
async def test_expand_query_hyde_empty_or_whitespace():
    assert await expand_query_hyde("") == ""
    assert await expand_query_hyde("   ") == "   "
    assert await expand_query_hyde(None) is None


@pytest.mark.asyncio
async def test_expand_query_hyde_offline_client_none_fallback():
    """Verifies that in an offline environment without Vertex credentials, raw query is returned immediately."""
    with patch("coordinator.get_vertex_client", return_value=None):
        raw_query = "how do airplanes fly"
        expanded = await expand_query_hyde(raw_query, client=None)
        assert expanded == raw_query


@pytest.mark.asyncio
async def test_expand_query_hyde_api_exception_fallback():
    """Verifies zero-exception graceful fallback to raw query when Vertex AI raises an exception."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("Vertex AI Quota Exceeded"))

    raw_query = "explain quantum entanglement"
    expanded = await expand_query_hyde(raw_query, client=mock_client)
    assert expanded == raw_query


@pytest.mark.asyncio
async def test_expand_query_hyde_timeout_fallback():
    """Verifies that slow or hanging LLM requests fall back to raw query after timeout."""
    mock_client = MagicMock()

    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(1.0)
        resp = MagicMock()
        resp.text = "Too slow expansion"
        return resp

    mock_client.aio.models.generate_content = AsyncMock(side_effect=slow_generate)

    raw_query = "database sharding strategies"
    # Set short timeout of 0.05 seconds
    expanded = await expand_query_hyde(raw_query, client=mock_client, timeout=0.05)
    assert expanded == raw_query


@pytest.mark.asyncio
async def test_expand_query_hyde_success():
    """Verifies that successful LLM generation returns the hypothetical expanded passage."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "   Aircraft generate aerodynamic lift via airfoils creating pressure differentials...   "
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    raw_query = "aerodynamic lift mechanism"
    expanded = await expand_query_hyde(raw_query, client=mock_client)
    assert expanded == "Aircraft generate aerodynamic lift via airfoils creating pressure differentials..."


@pytest.mark.asyncio
async def test_distributed_search_with_hyde_enabled():
    """Verifies that when use_hyde=True, the coordinator expands query and sends expanded text to compute shards."""
    coordinator._DNS_CACHE = {'nodes': ['http://node1:8000'], 'expiry': time.time() + 100}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        'results': [
            {'document': 'Aerodynamic lift document', 'score': 0.92}
        ]
    }

    posted_json = {}

    async def mock_post(url, **kwargs):
        nonlocal posted_json
        posted_json = kwargs.get('json', {})
        return mock_resp

    with patch('coordinator.expand_query_hyde', return_value="Hypothetical expanded text about aircraft lift"), \
         patch('httpx.AsyncClient.post', side_effect=mock_post):
        query = SearchQuery(text="how planes fly", top_k=1, use_hyde=True)
        response = await distributed_search(query)

        assert response['query'] == "how planes fly"
        assert response['search_text'] == "Hypothetical expanded text about aircraft lift"
        assert response['hyde_applied'] is True
        assert len(response['results']) == 1
        assert posted_json['text'] == "Hypothetical expanded text about aircraft lift"


@pytest.mark.asyncio
async def test_distributed_search_with_hyde_disabled():
    """Verifies that when use_hyde=False (default), HyDE is bypassed completely."""
    coordinator._DNS_CACHE = {'nodes': ['http://node1:8000'], 'expiry': time.time() + 100}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        'results': [
            {'document': 'Direct doc match', 'score': 0.88}
        ]
    }

    posted_json = {}

    async def mock_post(url, **kwargs):
        nonlocal posted_json
        posted_json = kwargs.get('json', {})
        return mock_resp

    with patch('coordinator.expand_query_hyde') as mock_hyde, \
         patch('httpx.AsyncClient.post', side_effect=mock_post):
        query = SearchQuery(text="raw query text", top_k=1, use_hyde=False)
        response = await distributed_search(query)

        mock_hyde.assert_not_called()
        assert response['query'] == "raw query text"
        assert response['search_text'] == "raw query text"
        assert response['hyde_applied'] is False
        assert posted_json['text'] == "raw query text"
