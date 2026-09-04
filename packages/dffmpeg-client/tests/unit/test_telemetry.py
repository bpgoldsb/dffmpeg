from unittest.mock import AsyncMock, patch

import pytest
from dffmpeg.client.config import ClientConfig
from dffmpeg.client.telemetry import (
    push_fallback_event,
    push_heartbeat,
    push_invocation_outcome,
)


def _config(**overrides):
    defaults = dict(
        client_id="test-client",
        hmac_key="secret",
        pushgateway_url="http://pushgateway.invalid:9091",
        telemetry_timeout=0.2,
    )
    defaults.update(overrides)
    return ClientConfig(**defaults)


@pytest.mark.anyio
async def test_push_disabled_makes_no_network_call():
    config = _config(pushgateway_enabled=False)

    with patch("dffmpeg.client.telemetry.httpx.AsyncClient") as mock_cls:
        await push_heartbeat(config)

    mock_cls.assert_not_called()


@pytest.mark.anyio
async def test_push_heartbeat_swallows_connection_errors():
    config = _config()

    with patch("dffmpeg.client.telemetry.httpx.AsyncClient", side_effect=OSError("connection refused")):
        # Must not raise -- a Pushgateway outage can never fail the caller (JXC-8).
        await push_heartbeat(config)


@pytest.mark.anyio
async def test_push_swallows_non_2xx_response():
    config = _config()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=AsyncMock(status_code=500))
    mock_http.__aenter__.return_value = mock_http
    mock_http.__aexit__.return_value = False

    with patch("dffmpeg.client.telemetry.httpx.AsyncClient", return_value=mock_http):
        # A 500 from Pushgateway itself must not raise either.
        await push_heartbeat(config)

    mock_http.post.assert_called_once()


@pytest.mark.anyio
async def test_push_invocation_outcome_success_path_single_push():
    config = _config()
    calls = []

    async def fake_push(cfg, job, instance, body, method="POST"):
        calls.append((job, instance, method, body))

    with patch("dffmpeg.client.telemetry._push", fake_push):
        await push_invocation_outcome(config, path="dffmpeg")

    assert len(calls) == 1
    job, instance, method, body = calls[0]
    assert job == "dffmpeg_client"
    assert instance == "test-client"
    assert method == "POST"
    assert 'path="dffmpeg"' in body


@pytest.mark.anyio
async def test_push_invocation_outcome_fallback_also_fires_fallback_event():
    config = _config()
    calls = []

    async def fake_push(cfg, job, instance, body, method="POST"):
        calls.append((job, instance, method, body))

    with patch("dffmpeg.client.telemetry._push", fake_push):
        await push_invocation_outcome(config, path="local_fallback", reason="no_workers_online")

    assert len(calls) == 2

    outcome_job, outcome_instance, _, outcome_body = calls[0]
    assert outcome_job == "dffmpeg_client"
    assert outcome_instance == "test-client"
    assert 'path="local_fallback"' in outcome_body

    fallback_job, fallback_instance, fallback_method, fallback_body = calls[1]
    assert fallback_job == "dffmpeg_client_fallback"
    # Each fallback event gets its OWN grouping key so Pushgateway's
    # replace-on-push semantics never silently drop a prior fallback.
    assert fallback_instance.startswith("test-client.")
    assert fallback_instance != "test-client"
    assert fallback_method == "PUT"
    assert 'reason="no_workers_online"' in fallback_body
    assert "dffmpeg_client_fallback_total" in fallback_body


@pytest.mark.anyio
async def test_push_fallback_event_unique_instance_per_call():
    config = _config()
    instances = []

    async def fake_push(cfg, job, instance, body, method="POST"):
        instances.append(instance)

    with patch("dffmpeg.client.telemetry._push", fake_push):
        await push_fallback_event(config, reason="coordinator_unreachable")
        await push_fallback_event(config, reason="coordinator_unreachable")

    assert len(instances) == 2
    assert instances[0] != instances[1]


@pytest.mark.anyio
async def test_push_wires_configured_telemetry_timeout_into_httpx_client():
    """
    JXC-8 requires the Pushgateway push to be bounded by its OWN short,
    separately-configured timeout. Verified here at the wiring level (the
    configured value reaches httpx.AsyncClient's constructor); separately,
    manual testing against a socket that accepts-but-never-responds confirms
    httpx.ReadTimeout actually fires at that bound in practice (see work item
    acceptance criteria) -- a real hung-server unit test is avoided here since
    it depends on cancelling a still-sleeping server task at teardown, which
    is exactly the kind of flaky, slow-to-clean-up test this package's own
    conventions (fast, mocked unit tests) steer away from.
    """
    config = _config(telemetry_timeout=1.5)

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=AsyncMock(status_code=200))
    mock_http.__aenter__.return_value = mock_http
    mock_http.__aexit__.return_value = False

    with patch("dffmpeg.client.telemetry.httpx.AsyncClient", return_value=mock_http) as mock_cls:
        await push_heartbeat(config)

    mock_cls.assert_called_once_with(timeout=1.5)
