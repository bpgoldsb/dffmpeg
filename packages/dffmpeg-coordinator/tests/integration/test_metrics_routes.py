import ipaddress

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import pytest

from dffmpeg.common.auth.request_signer import RequestSigner
from dffmpeg.coordinator.api import create_app
from dffmpeg.coordinator.config import CoordinatorConfig
from dffmpeg.coordinator.db import DBConfig

PROMETHEUS_CONTENT_TYPE_PREFIX = "text/plain; version=0.0.4"


def get_test_config(web_enabled: bool, db_path: str):
    repo_config = {"engine": "sqlite", "path": db_path}
    db_config = DBConfig(
        repositories={
            "auth": repo_config,
            "jobs": repo_config,
            "messages": repo_config,
            "workers": repo_config,
        }
    )
    return CoordinatorConfig(web_dashboard_enabled=web_enabled, database=db_config)


def test_metrics_endpoint_allowed(tmp_path):
    config = get_test_config(True, str(tmp_path / "test.db"))
    config.trusted_proxies = ["testclient"]
    config.allowed_metrics_ips = [ipaddress.ip_network("127.0.0.1/32")]

    app = create_app(config=config)
    with TestClient(app) as client:
        # Test client from 127.0.0.1 should be allowed
        response = client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(PROMETHEUS_CONTENT_TYPE_PREFIX)

        body = response.text
        # Prometheus exposition format: HELP/TYPE headers plus the sample lines.
        assert "# TYPE dffmpeg_workers_online gauge" in body
        assert "dffmpeg_workers_online 0" in body
        assert "# TYPE dffmpeg_jobs gauge" in body
        assert 'dffmpeg_jobs{window="current"} 0' in body


def test_metrics_endpoint_forbidden(tmp_path):
    config = get_test_config(True, str(tmp_path / "test.db"))
    config.trusted_proxies = ["testclient"]
    config.allowed_metrics_ips = [ipaddress.ip_network("127.0.0.1/32")]

    app = create_app(config=config)
    with TestClient(app) as client:
        # Test client from 192.168.1.5 should be forbidden
        response = client.get("/metrics", headers={"X-Forwarded-For": "192.168.1.5"})
        assert response.status_code == 403


def test_metrics_dashboard_disabled(tmp_path):
    # Metrics should still work even if dashboard is disabled
    config = get_test_config(False, str(tmp_path / "test.db"))
    config.trusted_proxies = ["testclient"]
    config.allowed_metrics_ips = [ipaddress.ip_network("127.0.0.1/32")]

    app = create_app(config=config)
    with TestClient(app) as client:
        response = client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
        assert response.status_code == 200


@pytest.mark.anyio
async def test_metrics_endpoint_reflects_online_worker_count(test_app, create_auth_identity, create_worker_record):
    """JXC-10: the online-worker gauge reflects worker_repo.get_workers_by_status('online')."""
    async with test_app.router.lifespan_context(test_app):
        await create_auth_identity(test_app, "worker01", "worker", RequestSigner.generate_key())
        await create_auth_identity(test_app, "worker02", "worker", RequestSigner.generate_key())

        # One online, one offline -- the gauge should count only the online one.
        await create_worker_record(test_app, "worker01", status="online")
        await create_worker_record(test_app, "worker02", status="offline")

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200

            body = response.text
            assert "dffmpeg_workers_online 1" in body
            # Both worker identities still appear in the by-worker job breakdown
            # (identity-seeded, per R4), even though only one is online.
            assert 'dffmpeg_jobs_by_worker{worker="worker01",window="current"} 0' in body
            assert 'dffmpeg_jobs_by_worker{worker="worker02",window="current"} 0' in body


@pytest.mark.anyio
async def test_metrics_endpoint_zero_online_workers(test_app):
    """JXC-10: with no workers registered at all, the gauge reads 0."""
    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200
            assert "dffmpeg_workers_online 0" in response.text
