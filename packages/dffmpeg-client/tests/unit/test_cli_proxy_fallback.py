"""
Tests for the dffmpeg_proxy path-selection / pre-output-fallback / timeout
logic added to cli.py:

* failsafe_force_local reads straight from ClientConfig and short-circuits
  before dffmpeg is ever attempted (JXC-1, JXC-2).
* _submit_and_await_first_signal classifies every pre-output failure mode
  (coordinator unreachable, the pinned no_workers_online 503, an
  unrecognized non-2xx response, and both timeout cases) into
  PreOutputFailure (JXC-5, JXC-13, JXC-14).
* proxy_main wires all of that into an actual local-binary exec, and never
  falls back once execution has produced a first sign of life (JXC-6).
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from dffmpeg.client import cli
from dffmpeg.client.cli import (
    PreOutputFailure,
    _is_no_workers_online,
    _submit_and_await_first_signal,
    stream_and_wait,
)
from dffmpeg.client.config import ClientConfig
from dffmpeg.common.models import (
    JobLogsMessage,
    JobLogsPayload,
    JobRecord,
    JobStatusMessage,
    JobStatusPayload,
    LogEntry,
)
from ulid import ULID


def _config(**overrides):
    defaults = dict(client_id="client1", hmac_key="secret", paths={}, pushgateway_enabled=False)
    defaults.update(overrides)
    return ClientConfig(**defaults)


def _job_record(job_id=None, **overrides):
    defaults = dict(
        job_id=job_id or ULID(),
        requester_id="client1",
        binary_name="ffmpeg",
        arguments=[],
        status="pending",
        transport="http_polling",
        transport_metadata={},
        heartbeat_interval=5,
        monitor=True,
    )
    defaults.update(overrides)
    return JobRecord(**defaults)


def _status_message(job_id, status, exit_code=None):
    return JobStatusMessage(
        recipient_id="client1",
        job_id=job_id,
        payload=JobStatusPayload(status=status, exit_code=exit_code),
    )


def _logs_message(job_id):
    return JobLogsMessage(
        recipient_id="client1",
        job_id=job_id,
        payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="hello")]),
    )


async def _agen_from(messages):
    for m in messages:
        yield m


async def _never_settles():
    await asyncio.sleep(3600)
    yield None  # pragma: no cover - unreachable


def _mock_client(config):
    client = MagicMock()
    client.config = config
    client._start_heartbeat_loop = AsyncMock()
    return client


# --------------------------------------------------------------------------
# _is_no_workers_online
# --------------------------------------------------------------------------


def _http_status_error(status_code, json_body=None):
    request = httpx.Request("POST", "http://coordinator.invalid/jobs/submit")
    if json_body is not None:
        response = httpx.Response(status_code, json=json_body, request=request)
    else:
        response = httpx.Response(status_code, content=b"not json", request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_is_no_workers_online_matches_pinned_shape():
    exc = _http_status_error(503, {"error": "no_workers_online", "detail": "no workers registered"})
    assert _is_no_workers_online(exc) is True


def test_is_no_workers_online_wrong_status():
    exc = _http_status_error(500, {"error": "no_workers_online"})
    assert _is_no_workers_online(exc) is False


def test_is_no_workers_online_wrong_body():
    exc = _http_status_error(503, {"error": "something_else"})
    assert _is_no_workers_online(exc) is False


def test_is_no_workers_online_unparseable_body_is_tolerant():
    exc = _http_status_error(503, json_body=None)
    assert _is_no_workers_online(exc) is False


# --------------------------------------------------------------------------
# _submit_and_await_first_signal: pre-output failure classification
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_submit_success_returns_job_and_first_message():
    config = _config()
    client = _mock_client(config)
    job = _job_record()
    client.submit_job = AsyncMock(return_value=job)
    first = _status_message(job.job_id, "running")
    client.stream_job = MagicMock(return_value=_agen_from([first]))

    returned_job, first_message, agen = await _submit_and_await_first_signal(client, "ffmpeg", ["-i", "in.mp4"], 5.0)

    assert returned_job is job
    assert first_message is first
    client.submit_job.assert_called_once()
    client._start_heartbeat_loop.assert_called_once_with(str(job.job_id), job.heartbeat_interval)


@pytest.mark.anyio
async def test_submit_coordinator_unreachable_is_pre_output_failure():
    config = _config()
    client = _mock_client(config)
    client.submit_job = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 5.0)

    assert excinfo.value.reason.startswith("coordinator_unreachable")


@pytest.mark.anyio
async def test_submit_no_workers_online_is_distinctly_classified():
    config = _config()
    client = _mock_client(config)
    client.submit_job = AsyncMock(
        side_effect=_http_status_error(503, {"error": "no_workers_online", "detail": "0 workers online"})
    )

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 5.0)

    # Positively identified, not folded into the generic http-error bucket.
    assert excinfo.value.reason == "no_workers_online"


@pytest.mark.anyio
async def test_submit_unrecognized_non_2xx_still_falls_back():
    config = _config()
    client = _mock_client(config)
    client.submit_job = AsyncMock(side_effect=_http_status_error(500, {"error": "internal_error"}))

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 5.0)

    assert excinfo.value.reason == "coordinator_http_error_500"


@pytest.mark.anyio
async def test_submit_timeout_is_pre_output_failure():
    config = _config()
    client = _mock_client(config)

    async def _hang(*a, **kw):
        await asyncio.sleep(3600)

    client.submit_job = AsyncMock(side_effect=_hang)

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 0.05)

    assert excinfo.value.reason == "coordinator_timeout_on_submit"


@pytest.mark.anyio
async def test_wedged_worker_after_submit_success_times_out_and_falls_back():
    """
    A worker registered 'online' but wedged: the coordinator happily accepts
    the job, but nothing at all ever comes back over the stream (JXC-14).
    """
    config = _config()
    client = _mock_client(config)
    job = _job_record()
    client.submit_job = AsyncMock(return_value=job)
    client.stream_job = MagicMock(return_value=_never_settles())

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 0.05)

    assert excinfo.value.reason == "coordinator_timeout_no_execution_signal"


@pytest.mark.anyio
async def test_unexpected_error_before_first_signal_is_still_pre_output_failure():
    config = _config()
    client = _mock_client(config)
    client.submit_job = AsyncMock(side_effect=ValueError("something truly unexpected"))

    with pytest.raises(PreOutputFailure) as excinfo:
        await _submit_and_await_first_signal(client, "ffmpeg", [], 5.0)

    assert excinfo.value.reason.startswith("unexpected_error:ValueError")


# --------------------------------------------------------------------------
# stream_and_wait: replaying a pre-fetched first message
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stream_and_wait_replays_first_message_then_continues_same_generator():
    config = _config()
    client = _mock_client(config)
    client.cancel_job = AsyncMock()
    job_id = str(ULID())

    first = _logs_message(job_id)
    second = _status_message(job_id, "completed", exit_code=0)
    agen = _agen_from([second])

    exit_code = await stream_and_wait(client, job_id, "http_polling", {}, first_message=first, agen=agen)

    assert exit_code == 0


# --------------------------------------------------------------------------
# proxy_main: end-to-end wiring
# --------------------------------------------------------------------------


class _FakeAsyncClient:
    """Minimal async-context-manager stand-in for DFFmpegClient."""

    def __init__(self, config):
        self.config = config
        self._start_heartbeat_loop = AsyncMock()
        self.cancel_job = AsyncMock()
        self.submit_job = AsyncMock()
        self.stream_job = MagicMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _run_proxy_main(monkeypatch, argv, config, fake_client, exec_calls):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "DFFmpegClient", lambda cfg: fake_client)

    def fake_execv(path, full_argv):
        exec_calls.append((path, list(full_argv)))
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execv", fake_execv)


def test_proxy_main_failsafe_forces_local_without_attempting_dffmpeg(monkeypatch):
    config = _config(failsafe_force_local=True, local_ffmpeg_path="/usr/lib/jellyfin-ffmpeg/ffmpeg")
    exec_calls = []

    client_constructed = []
    monkeypatch.setattr(sys, "argv", ["ffmpeg", "-i", "in.mp4", "out.m3u8"])
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "DFFmpegClient", lambda cfg: client_constructed.append(cfg))

    def fake_execv(path, full_argv):
        exec_calls.append((path, list(full_argv)))
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execv", fake_execv)

    with pytest.raises(SystemExit):
        cli.proxy_main()

    assert exec_calls == [
        ("/usr/lib/jellyfin-ffmpeg/ffmpeg", ["/usr/lib/jellyfin-ffmpeg/ffmpeg", "-i", "in.mp4", "out.m3u8"])
    ]
    # dffmpeg must never be attempted at all when the failsafe is set.
    assert client_constructed == []


def test_proxy_main_falls_back_on_coordinator_unreachable(monkeypatch):
    config = _config(coordinator_request_timeout=5.0)
    fake_client = _FakeAsyncClient(config)
    fake_client.submit_job = AsyncMock(side_effect=httpx.ConnectError("no route to host"))
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    with pytest.raises(SystemExit):
        cli.proxy_main()

    assert len(exec_calls) == 1
    path, full_argv = exec_calls[0]
    assert path == config.local_ffmpeg_path
    assert full_argv == [config.local_ffmpeg_path, "-i", "in.mp4", "out.m3u8"]


def test_proxy_main_falls_back_on_pinned_no_workers_online_response(monkeypatch):
    config = _config(coordinator_request_timeout=5.0)
    fake_client = _FakeAsyncClient(config)
    fake_client.submit_job = AsyncMock(
        side_effect=_http_status_error(503, {"error": "no_workers_online", "detail": "0 online"})
    )
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    with pytest.raises(SystemExit):
        cli.proxy_main()

    assert len(exec_calls) == 1
    assert exec_calls[0][0] == config.local_ffmpeg_path


def test_proxy_main_falls_back_on_wedged_worker_timeout(monkeypatch):
    config = _config(coordinator_request_timeout=0.05)
    fake_client = _FakeAsyncClient(config)
    fake_client.submit_job = AsyncMock(return_value=_job_record())
    fake_client.stream_job = MagicMock(return_value=_never_settles())
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    with pytest.raises(SystemExit):
        cli.proxy_main()

    assert len(exec_calls) == 1
    assert exec_calls[0][0] == config.local_ffmpeg_path


def test_proxy_main_does_not_fall_back_after_execution_has_started(monkeypatch):
    """
    Mid-stream failure (job already showed a sign of life, then errors) is
    explicitly out of scope (JXC-6) -- this must return a plain failure exit
    code, never exec the local binary.
    """
    config = _config(coordinator_request_timeout=5.0)
    fake_client = _FakeAsyncClient(config)
    job = _job_record()
    fake_client.submit_job = AsyncMock(return_value=job)

    async def _agen():
        yield _logs_message(str(job.job_id))
        raise RuntimeError("connection dropped mid-stream")

    fake_client.stream_job = MagicMock(return_value=_agen())
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    with pytest.raises(SystemExit) as excinfo:
        cli.proxy_main()

    assert exec_calls == []
    assert excinfo.value.code == 1


def test_proxy_main_success_execs_nothing_and_exits_with_job_code(monkeypatch):
    config = _config(coordinator_request_timeout=5.0)
    fake_client = _FakeAsyncClient(config)
    job = _job_record()
    fake_client.submit_job = AsyncMock(return_value=job)
    fake_client.stream_job = MagicMock(return_value=_agen_from([_status_message(job.job_id, "completed", 0)]))
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    with pytest.raises(SystemExit) as excinfo:
        cli.proxy_main()

    assert exec_calls == []
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------
# Telemetry call is fire-and-forget from proxy_main's sync tail
# --------------------------------------------------------------------------


def test_push_telemetry_sync_never_raises_even_if_push_fails():
    config = _config()

    async def _boom(*a, **kw):
        raise RuntimeError("pushgateway is on fire")

    with patch("dffmpeg.client.cli.push_invocation_outcome", _boom):
        cli._push_telemetry_sync(config, path="local_fallback", reason="coordinator_unreachable")


def test_proxy_main_fallback_pushes_telemetry_with_reason(monkeypatch):
    config = _config(coordinator_request_timeout=5.0)
    fake_client = _FakeAsyncClient(config)
    fake_client.submit_job = AsyncMock(side_effect=httpx.ConnectError("boom"))
    exec_calls = []

    _run_proxy_main(monkeypatch, ["ffmpeg", "-i", "in.mp4", "out.m3u8"], config, fake_client, exec_calls)

    telemetry_calls = []

    async def fake_push_invocation_outcome(cfg, path, reason=None):
        telemetry_calls.append((path, reason))

    monkeypatch.setattr(cli, "push_invocation_outcome", fake_push_invocation_outcome)

    with pytest.raises(SystemExit):
        cli.proxy_main()

    assert len(telemetry_calls) == 1
    path, reason = telemetry_calls[0]
    assert path == "local_fallback"
    assert reason.startswith("coordinator_unreachable")
