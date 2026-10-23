from unittest.mock import ANY, AsyncMock, patch

import pytest
from ulid import ULID

from dffmpeg.coordinator.config import JanitorConfig
from dffmpeg.coordinator.db.jobs import JobRecord
from dffmpeg.coordinator.db.workers import WorkerRecord
from dffmpeg.coordinator.janitor import Janitor


@pytest.fixture
def worker_repo():
    return AsyncMock()


@pytest.fixture
def job_repo():
    return AsyncMock()


@pytest.fixture
def transports():
    return AsyncMock()


@pytest.fixture
def config():
    return JanitorConfig()


@pytest.fixture
def janitor(worker_repo, job_repo, transports, config):
    return Janitor(worker_repo, job_repo, transports, config)


@pytest.mark.anyio
async def test_reap_workers(janitor, worker_repo):
    worker = WorkerRecord(
        worker_id="w1", status="online", registration_interval=10, transport="http", transport_metadata={}
    )
    # Populate fields that should be cleared
    worker.capabilities = ["h264"]
    worker.binaries = ["ffmpeg"]
    worker.paths = ["/tmp"]

    worker_repo.get_stale_workers.return_value = [worker]

    await janitor.reap_workers()

    # Check if worker status was updated to offline
    assert worker.status == "offline"
    # Check if fields were cleared
    assert worker.capabilities == []
    assert worker.binaries == []
    assert worker.paths == []
    assert worker.transport == "none"
    assert worker.transport_metadata == {}
    assert worker.registration_interval == 0

    worker_repo.add_or_update.assert_called_once_with(worker)


@pytest.mark.anyio
async def test_reap_draining_workers(janitor, worker_repo):
    worker = WorkerRecord(
        worker_id="w1", status="draining", registration_interval=10, transport="http", transport_metadata={}
    )
    worker.capabilities = ["h264"]
    worker.binaries = ["ffmpeg"]
    worker.paths = ["/tmp"]

    worker_repo.get_stale_workers.return_value = [worker]

    await janitor.reap_workers()

    # Check if draining worker is successfully marked offline when stale
    assert worker.status == "offline"
    assert worker.capabilities == []
    assert worker.binaries == []
    assert worker.paths == []
    assert worker.transport == "none"
    assert worker.transport_metadata == {}
    assert worker.registration_interval == 0

    worker_repo.add_or_update.assert_called_once_with(worker)


@pytest.mark.anyio
async def test_reap_running_jobs(janitor, job_repo, transports):
    job = JobRecord(
        job_id=ULID(),
        requester_id="client1",
        worker_id="worker1",
        binary_name="ffmpeg",
        status="running",
        transport="http",
        transport_metadata={},
    )
    job_repo.get_stale_running_jobs.return_value = [job]
    job_repo.update_status.return_value = True

    await janitor.reap_running_jobs()

    job_repo.update_status.assert_called_once_with(job.job_id, "failed", previous_status="running", timestamp=ANY)

    # Check notifications
    assert transports.send_message.call_count == 2
    # First one to client, second to worker (order in code)
    # Actually order depends on implementation, but both should be called.


@pytest.mark.anyio
async def test_reap_running_jobs_update_fails(janitor, job_repo, transports):
    job = JobRecord(
        job_id=ULID(),
        requester_id="client1",
        worker_id="worker1",
        binary_name="ffmpeg",
        status="running",
        transport="http",
        transport_metadata={},
    )
    job_repo.get_stale_running_jobs.return_value = [job]
    job_repo.update_status.return_value = False  # Simulate race condition failure

    await janitor.reap_running_jobs()

    # Should NOT send notifications
    transports.send_message.assert_not_called()


@pytest.mark.anyio
async def test_reap_assigned_jobs(janitor, job_repo, transports):
    job = JobRecord(
        job_id=ULID(),
        requester_id="client1",
        worker_id="worker1",
        binary_name="ffmpeg",
        status="assigned",
        transport="http",
        transport_metadata={},
    )
    job_repo.get_stale_assigned_jobs.return_value = [job]
    job_repo.update_status.return_value = True

    await janitor.reap_assigned_jobs()

    job_repo.update_status.assert_called_once_with(job.job_id, "pending", previous_status="assigned", timestamp=ANY)

    # Check notifications (only worker)
    assert transports.send_message.call_count == 1


@pytest.mark.anyio
async def test_reap_pending_jobs(janitor, job_repo, transports):
    with patch("dffmpeg.coordinator.janitor.process_job_assignment", new_callable=AsyncMock) as mock_process:
        # Job 1: Retry (10s old)
        job1 = JobRecord(
            job_id=ULID(),
            requester_id="c1",
            binary_name="ffmpeg",
            status="pending",
            transport="http",
            transport_metadata={},
        )

        # Job 2: Fail (40s old)
        job2 = JobRecord(
            job_id=ULID(),
            requester_id="c1",
            binary_name="ffmpeg",
            status="pending",
            transport="http",
            transport_metadata={},
        )

        # Setup job repo returns
        # get_stale_pending_jobs called twice:
        # 1. min=5, max=30 -> returns [job1]
        # 2. min=30 -> returns [job2]
        job_repo.get_stale_pending_jobs.side_effect = [[job1], [job2]]

        job_repo.update_status.return_value = True

        await janitor.reap_pending_jobs()

        # Verify process_job_assignment called for job1
        mock_process.assert_called_once()
        assert mock_process.call_args[0][0] == job1.job_id

        # Verify fail logic for job2
        job_repo.update_status.assert_called_once_with(job2.job_id, "failed", previous_status="pending", timestamp=ANY)

        # Verify notification for job2
        assert transports.send_message.call_count == 1
        assert transports.send_message.call_args[0][0].job_id == job2.job_id


@pytest.mark.anyio
async def test_reap_abandoned_monitored_jobs(janitor, job_repo, transports):
    job = JobRecord(
        job_id=ULID(),
        requester_id="client1",
        worker_id="worker1",
        binary_name="ffmpeg",
        status="running",
        monitor=True,
        transport="http",
        transport_metadata={},
    )
    job_repo.get_stale_monitored_jobs.return_value = [job]
    job_repo.update_status.return_value = True

    await janitor.reap_abandoned_monitored_jobs()

    job_repo.update_status.assert_called_once_with(job.job_id, "canceling", timestamp=ANY)

    # Check notifications: 1 to client, 1 to worker
    assert transports.send_message.call_count == 2
    # Verify both recipients got a message
    recipients = [call.args[0].recipient_id for call in transports.send_message.call_args_list]
    assert "client1" in recipients
    assert "worker1" in recipients


@pytest.fixture
def message_repo():
    repo = AsyncMock()
    repo.delete_expired.return_value = (0, None)
    return repo


@pytest.fixture
def pruning_janitor(worker_repo, job_repo, transports, config, message_repo):
    return Janitor(worker_repo, job_repo, transports, config, message_repo=message_repo)


@pytest.mark.anyio
async def test_reap_old_messages_skipped_without_repo(janitor):
    await janitor.reap_old_messages()  # no message_repo: must not raise


@pytest.mark.anyio
async def test_reap_old_messages_passes_config(pruning_janitor, message_repo):
    from datetime import timedelta

    await pruning_janitor.reap_old_messages()
    message_repo.delete_expired.assert_awaited_once_with(
        message_retention=timedelta(days=2),
        job_logs_retention=timedelta(days=14),
        last_message_id=None,
        batch_size=5000,
    )


@pytest.mark.anyio
async def test_reap_old_messages_full_batch_runs_next_call_with_cursor(pruning_janitor, message_repo):
    message_repo.delete_expired.side_effect = [(5000, "ID1"), (5000, "ID2"), (3, "ID3")]
    for _ in range(3):
        await pruning_janitor.reap_old_messages()
    cursors = [c.kwargs["last_message_id"] for c in message_repo.delete_expired.await_args_list]
    assert cursors == [None, "ID1", "ID2"]
    # short batch: cursor reset and throttled for the hour
    assert pruning_janitor._message_prune_cursor is None
    await pruning_janitor.reap_old_messages()
    assert message_repo.delete_expired.await_count == 3


@pytest.mark.anyio
async def test_reap_old_messages_short_batch_throttles_for_an_hour(pruning_janitor, message_repo):
    message_repo.delete_expired.return_value = (10, "ID")
    await pruning_janitor.reap_old_messages()
    await pruning_janitor.reap_old_messages()
    assert message_repo.delete_expired.await_count == 1
    pruning_janitor._next_message_prune = 0  # the hour elapses
    await pruning_janitor.reap_old_messages()
    assert message_repo.delete_expired.await_count == 2


@pytest.mark.anyio
async def test_reap_old_messages_exception_is_not_a_full_batch(pruning_janitor, message_repo):
    message_repo.delete_expired.side_effect = RuntimeError("db down")
    with pytest.raises(RuntimeError):
        await pruning_janitor.reap_old_messages()
    message_repo.delete_expired.side_effect = None
    await pruning_janitor.reap_old_messages()
    assert message_repo.delete_expired.await_count == 1  # throttled


@pytest.mark.anyio
async def test_run_all_prunes_messages_last(pruning_janitor):
    order = []
    for name in (
        "reap_workers",
        "reap_running_jobs",
        "reap_assigned_jobs",
        "reap_pending_jobs",
        "reap_abandoned_monitored_jobs",
        "reap_old_messages",
    ):
        setattr(pruning_janitor, name, AsyncMock(side_effect=lambda n=name: order.append(n)))
    await pruning_janitor.run_all()
    assert order[-1] == "reap_old_messages" and len(order) == 6


@pytest.mark.anyio
async def test_named_tasks_do_not_prune(pruning_janitor, message_repo):
    for name in ("reap_workers", "reap_running_jobs", "reap_assigned_jobs", "reap_pending_jobs"):
        setattr(pruning_janitor, name, AsyncMock())
    pruning_janitor.reap_abandoned_monitored_jobs = AsyncMock()
    await pruning_janitor.start(schedule_task=False)
    pruning_janitor.schedule_task("clean_workers")
    pruning_janitor.schedule_task("clean_jobs")
    await pruning_janitor._queue.join()
    await pruning_janitor.stop()
    message_repo.delete_expired.assert_not_awaited()
