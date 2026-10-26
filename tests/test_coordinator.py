import asyncio
import os
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
import coordinator


@pytest.mark.asyncio
async def test_get_active_compute_nodes_default():
    coordinator._DNS_CACHE = {'nodes': [], 'expiry': 0.0}
    with patch.dict(os.environ, {}, clear=True):
        nodes = await coordinator.get_active_compute_nodes()
        assert nodes == coordinator.COMPUTE_NODES


@pytest.mark.asyncio
async def test_get_active_compute_nodes_headless_dns_with_caching():
    coordinator._DNS_CACHE = {'nodes': [], 'expiry': 0.0}
    fake_env = {'COMPUTE_HEADLESS_HOST': 'compute-headless.ares.svc.cluster.local'}
    fake_dns = ('compute-headless', [], ['10.244.1.5', '10.244.2.8', '10.244.3.11'])

    with patch.dict(os.environ, fake_env), patch('socket.gethostbyname_ex', return_value=fake_dns) as mock_dns:
        nodes = await coordinator.get_active_compute_nodes()
        assert nodes == [
            'http://10.244.1.5:8000',
            'http://10.244.2.8:8000',
            'http://10.244.3.11:8000'
        ]
        assert mock_dns.call_count == 1

        # Second call within TTL must hit cache without invoking socket
        nodes_cached = await coordinator.get_active_compute_nodes()
        assert nodes_cached == nodes
        assert mock_dns.call_count == 1


@pytest.mark.asyncio
async def test_distributed_search_resilient_partial_failure():
    coordinator._DNS_CACHE = {'nodes': ['http://node1:8000', 'http://node2:8000'], 'expiry': time.time() + 100}

    mock_resp_node1 = MagicMock()
    mock_resp_node1.status_code = 200
    mock_resp_node1.json.return_value = {
        'results': [
            {'document': 'Doc A', 'score': 0.95},
            {'document': 'Doc B', 'score': 0.80}
        ]
    }

    async def mock_post(url, **kwargs):
        if 'node1' in url:
            return mock_resp_node1
        raise Exception('Connection refused by node2')

    with patch('httpx.AsyncClient.post', side_effect=mock_post):
        query = coordinator.SearchQuery(text='test query', top_k=2)
        response = await coordinator.distributed_search(query)

        assert len(response['results']) == 2
        assert response['results'][0]['document'] == 'Doc A'
        assert response['results'][0]['rank'] == 1
        assert response['results'][1]['document'] == 'Doc B'
        assert response['results'][1]['rank'] == 2

        health = response['cluster_health']
        assert health['total_shards'] == 2
        assert health['successful_shards'] == 1
        assert health['failed_shards'] == 1
        assert health['partial_results'] is True


@pytest.mark.asyncio
async def test_distributed_search_all_shards_fail_raises_503():
    coordinator._DNS_CACHE = {'nodes': ['http://node1:8000', 'http://node2:8000'], 'expiry': time.time() + 100}

    async def mock_post_fail(url, **kwargs):
        raise Exception('Cluster partition')

    with patch('httpx.AsyncClient.post', side_effect=mock_post_fail):
        query = coordinator.SearchQuery(text='test query', top_k=2)
        with pytest.raises(HTTPException) as exc_info:
            await coordinator.distributed_search(query)
        assert exc_info.value.status_code == 503
        assert 'All compute shards failed' in exc_info.value.detail


@pytest.mark.asyncio
async def test_distributed_ingest_batched():
    coordinator._DNS_CACHE = {'nodes': ['http://node1:8000', 'http://node2:8000'], 'expiry': time.time() + 100}

    posted_payloads = {}

    async def mock_post(url, **kwargs):
        json_data = kwargs.get('json', {})
        posted_payloads[url] = json_data.get('documents', [])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return mock_resp

    with patch('httpx.AsyncClient.post', side_effect=mock_post):
        docs = ['doc1', 'doc2', 'doc3', 'doc4']
        payload = coordinator.IngestPayload(documents=docs)
        resp = await coordinator.distributed_ingest(payload)

        assert resp['documents_routed'] == 4
        # Verify round-robin bucketing: doc1 and doc3 to node1, doc2 and doc4 to node2
        assert posted_payloads['http://node1:8000/ingest'] == ['doc1', 'doc3']
        assert posted_payloads['http://node2:8000/ingest'] == ['doc2', 'doc4']
