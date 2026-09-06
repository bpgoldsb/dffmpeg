import ipaddress
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from ulid import ULID

from dffmpeg.common.auth.request_signer import RequestSigner
from dffmpeg.common.models import JobRecord
from dffmpeg.coordinator.api import create_app
from dffmpeg.coordinator.config import CoordinatorConfig
from dffmpeg.coordinator.db import DBConfig

PROMETHEUS_CONTENT_TYPE_PREFIX = "text/plain; version=0.0.4"


async def _insert_job(
    app,
    *,
    status: str,
    created_at: datetime,
    last_update: datetime,
    binary_name: str = "ffmpeg",
) -> JobRecord:
    """
    Insert a job directly with explicit created_at/last_update, bypassing the
    create_job_record fixture (which always stamps both to "now") -- needed to
    exercise the last_update-vs-created_at bucketing distinction.
    """
    job = JobRecord(
        job_id=ULID(),
        requester_id="client01",
        binary_name=binary_name,
        arguments=["-i", "in", "out"],
        paths=["Movies"],
        status=status,
        transport="http_polling",
        created_at=created_at,
        last_update=last_update,
    )
    await app.state.db.jobs.create_job(job)
    return job


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

        # dffmpeg_jobs_succeeded: only last_1m/last_5m, never "current".
        assert "# TYPE dffmpeg_jobs_succeeded gauge" in body
        assert 'dffmpeg_jobs_succeeded{window="last_1m"} 0' in body
        assert 'dffmpeg_jobs_succeeded{window="last_5m"} 0' in body
        assert 'dffmpeg_jobs_succeeded{window="current"}' not in body


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


@pytest.mark.anyio
async def test_metrics_succeeded_counts_completed_job(test_app):
    """A completed job whose last_update is recent is counted in both windows."""
    async with test_app.router.lifespan_context(test_app):
        now = datetime.now(timezone.utc)
        await _insert_job(
            test_app,
            status="completed",
            created_at=now - timedelta(seconds=30),
            last_update=now - timedelta(seconds=10),
        )

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200
            body = response.text
            assert 'dffmpeg_jobs_succeeded{window="last_1m"} 1' in body
            assert 'dffmpeg_jobs_succeeded{window="last_5m"} 1' in body


@pytest.mark.anyio
async def test_metrics_succeeded_excludes_failed_job(test_app):
    """A failed job -- even one whose last_update is recent -- is excluded from
    the success-only bucket (acceptance #4)."""
    async with test_app.router.lifespan_context(test_app):
        now = datetime.now(timezone.utc)
        await _insert_job(
            test_app,
            status="failed",
            created_at=now - timedelta(seconds=30),
            last_update=now - timedelta(seconds=10),
        )

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200
            body = response.text
            assert 'dffmpeg_jobs_succeeded{window="last_1m"} 0' in body
            assert 'dffmpeg_jobs_succeeded{window="last_5m"} 0' in body
            # Sanity: the failed job IS still visible in the general-purpose
            # family, confirming the exclusion is specific to the success bucket.
            assert 'dffmpeg_jobs{window="last_1m"} 1' in body


@pytest.mark.anyio
async def test_metrics_succeeded_keyed_on_last_update_not_created_at(test_app):
    """
    A job created ~10 minutes ago (older than the 5-minute get_recent_jobs
    window) but completed within the last minute must still be counted in
    last_5m -- because get_recent_jobs bounds its candidate set on last_update,
    not created_at, and this bucket must key on the same field (acceptance #5).
    Bucketing on created_at instead would make this job invisible here.
    """
    async with test_app.router.lifespan_context(test_app):
        now = datetime.now(timezone.utc)
        await _insert_job(
            test_app,
            status="completed",
            created_at=now - timedelta(minutes=10),
            last_update=now - timedelta(seconds=30),
        )

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200
            body = response.text
            assert 'dffmpeg_jobs_succeeded{window="last_5m"} 1' in body
            # It completed <1m ago, so it also lands in last_1m.
            assert 'dffmpeg_jobs_succeeded{window="last_1m"} 1' in body


@pytest.mark.anyio
async def test_metrics_succeeded_render_format(test_app):
    """
    Rendering-format check standing in for a manual local run (acceptance #6):
    asserts the exact HELP/TYPE/sample lines for the new family in valid
    Prometheus text-exposition format, with no 'current' window emitted.
    """
    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
            assert response.status_code == 200
            lines = response.text.splitlines()

            help_idx = lines.index(
                "# HELP dffmpeg_jobs_succeeded "
                "Count of jobs that reached status=completed, bucketed by the trailing "
                "window in which their last_update (== completion time) falls. No "
                "'current' window is emitted -- a completed job is never current, so "
                "that series would be permanently zero."
            )
            # The family is exactly HELP, TYPE, then the two windowed samples in
            # declaration order -- no "current" sample sneaks in between.
            assert lines[help_idx : help_idx + 4] == [
                lines[help_idx],
                "# TYPE dffmpeg_jobs_succeeded gauge",
                'dffmpeg_jobs_succeeded{window="last_1m"} 0',
                'dffmpeg_jobs_succeeded{window="last_5m"} 0',
            ]
