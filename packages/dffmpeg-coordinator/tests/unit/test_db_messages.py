from datetime import datetime, timedelta, timezone

import pytest
from ulid import ULID

from dffmpeg.common.models import (
    JobLogsMessage,
    JobLogsPayload,
    JobStatusMessage,
    JobStatusPayload,
    LogEntry,
    VerifyRegistrationMessage,
    VerifyRegistrationPayload,
)
from dffmpeg.coordinator.db.messages.sqlite import SQLiteMessageRepository


@pytest.fixture
async def message_repo(tmp_path):
    db_path = tmp_path / "test_messages.db"
    repo = SQLiteMessageRepository(engine="sqlite", path=str(db_path))
    await repo.setup()
    return repo


@pytest.mark.anyio
async def test_get_job_messages_basic(message_repo):
    job_id = ULID()
    msg1 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="log 1")])
    )
    msg2 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="log 2")])
    )
    msg3 = JobLogsMessage(
        recipient_id="client1", job_id=ULID(), payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="other")])
    )

    await message_repo.add_message(msg1)
    await message_repo.add_message(msg2)
    await message_repo.add_message(msg3)

    messages = await message_repo.get_job_messages(job_id)
    assert len(messages) == 2
    assert messages[0].message_id == msg1.message_id
    assert messages[1].message_id == msg2.message_id


@pytest.mark.anyio
async def test_get_job_messages_type_filter(message_repo):
    job_id = ULID()
    msg1 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="log")])
    )
    msg2 = JobStatusMessage(recipient_id="client1", job_id=job_id, payload=JobStatusPayload(status="running"))

    await message_repo.add_message(msg1)
    await message_repo.add_message(msg2)

    messages = await message_repo.get_job_messages(job_id, message_type="job_logs")
    assert len(messages) == 1
    assert messages[0].message_type == "job_logs"
    assert messages[0].message_id == msg1.message_id


@pytest.mark.anyio
async def test_get_job_messages_since_id(message_repo):
    job_id = ULID()
    msg1 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="1")])
    )
    msg2 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="2")])
    )
    msg3 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="3")])
    )

    await message_repo.add_message(msg1)
    await message_repo.add_message(msg2)
    await message_repo.add_message(msg3)

    messages = await message_repo.get_job_messages(job_id, since_message_id=msg1.message_id)
    assert len(messages) == 2
    assert messages[0].message_id == msg2.message_id
    assert messages[1].message_id == msg3.message_id


@pytest.mark.anyio
async def test_get_job_messages_limit(message_repo):
    job_id = ULID()
    msg1 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="1")])
    )
    msg2 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="2")])
    )
    msg3 = JobLogsMessage(
        recipient_id="client1", job_id=job_id, payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="3")])
    )

    await message_repo.add_message(msg1)
    await message_repo.add_message(msg2)
    await message_repo.add_message(msg3)

    # Limit 2 should give us msg2 and msg3 (most recent)
    messages = await message_repo.get_job_messages(job_id, limit=2)
    assert len(messages) == 2
    assert messages[0].message_id == msg2.message_id
    assert messages[1].message_id == msg3.message_id


def _verify_msg(recipient_id, message_id=None):
    kwargs = {"message_id": message_id} if message_id is not None else {}
    return VerifyRegistrationMessage(
        recipient_id=recipient_id, payload=VerifyRegistrationPayload(registration_token="t"), **kwargs
    )


@pytest.mark.anyio
async def test_get_messages_pages_in_id_order(message_repo, monkeypatch):
    monkeypatch.setattr(type(message_repo), "POLL_PAGE_SIZE", 3)
    msgs = [_verify_msg("worker1") for _ in range(7)]
    # Insert out of ID order so rowid order and message_id order disagree.
    for msg in reversed(msgs):
        await message_repo.add_message(msg)

    seen = []
    first_page = await message_repo.retrieve_messages("worker1")
    seen.extend(first_page)
    cursor = first_page[-1].message_id
    while True:
        page = await message_repo.retrieve_messages("worker1", last_message_id=cursor)
        if not page:
            break
        assert len(page) <= 3
        seen.extend(page)
        cursor = page[-1].message_id

    assert [m.message_id for m in seen] == sorted(m.message_id for m in msgs)


@pytest.mark.anyio
async def test_fresh_poll_skips_stale_undelivered_orphans(message_repo):
    # The 2026-09-27 incident: a worker restarted with a 2-day-old undelivered orphan in the
    # table, adopted it as its cursor, and every later poll replayed everything since.
    stale_id = ULID.from_datetime(datetime.now(timezone.utc) - timedelta(days=2))
    orphan = _verify_msg("worker1", message_id=stale_id)
    fresh = _verify_msg("worker1")
    await message_repo.add_message(orphan)
    await message_repo.add_message(fresh)

    messages = await message_repo.get_messages("worker1")
    assert [m.message_id for m in messages] == [fresh.message_id]


@pytest.mark.anyio
async def test_fresh_job_poll_keeps_old_undelivered_messages(message_repo):
    # Job-scoped polls are not age-bounded: a long job's client may reconnect late.
    job_id = ULID()
    old_id = ULID.from_datetime(datetime.now(timezone.utc) - timedelta(hours=2))
    old = JobLogsMessage(
        message_id=old_id,
        recipient_id="client1",
        job_id=job_id,
        payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="old")]),
    )
    await message_repo.add_message(old)

    messages = await message_repo.get_messages("client1", job_id=job_id)
    assert [m.message_id for m in messages] == [old.message_id]


@pytest.mark.anyio
async def test_setup_creates_declared_indexes(message_repo):
    async with message_repo._connect() as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='messages'") as cur:
            names = {row[0] for row in await cur.fetchall()}
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM messages WHERE recipient_id = 'w' AND message_id > 'x' "
            "ORDER BY message_id LIMIT 100"
        ) as cur:
            plan = " ".join(str(row[-1]) for row in await cur.fetchall())

    assert {"ix_messages_recipient_id_message_id", "ix_messages_job_id_message_id"} <= names
    assert "ix_messages_recipient_id_message_id" in plan

    # Idempotent against an existing database.
    await message_repo.setup()


def _msg_at(message_cls, age: timedelta, recipient_id="client1", sent_at=None, **kwargs):
    """Builds a message whose ULID (and so its age for retention) is `age` old."""
    msg = message_cls(recipient_id=recipient_id, **kwargs)
    msg.message_id = ULID.from_datetime(datetime.now(timezone.utc) - age)
    msg.sent_at = sent_at
    return msg


def _status(age, **kw):
    return _msg_at(JobStatusMessage, age, job_id=ULID(), payload=JobStatusPayload(status="running"), **kw)


def _logs(age, **kw):
    return _msg_at(
        JobLogsMessage,
        age,
        job_id=ULID(),
        payload=JobLogsPayload(logs=[LogEntry(stream="stdout", content="x")]),
        **kw,
    )


RETENTION = dict(message_retention=timedelta(days=2), job_logs_retention=timedelta(days=14))


async def _remaining(repo):
    rows = await repo.get_rows("SELECT message_id FROM messages ORDER BY message_id", ())
    return {row["message_id"] for row in rows}


@pytest.mark.anyio
async def test_delete_expired_predicate(message_repo):
    old_status = _status(timedelta(days=2, minutes=1))
    young_status = _status(timedelta(days=1, hours=23))
    undelivered_old = _status(timedelta(days=30), sent_at=None)
    delivered_old = _status(timedelta(days=30), sent_at=datetime.now(timezone.utc))
    young_logs = _logs(timedelta(days=3))
    mid_logs = _logs(timedelta(days=13, hours=23))
    old_logs = _logs(timedelta(days=14, minutes=1))
    for m in (old_status, young_status, undelivered_old, delivered_old, young_logs, mid_logs, old_logs):
        await message_repo.add_message(m)

    deleted, last_id = await message_repo.delete_expired(**RETENTION)

    assert deleted == 4
    assert last_id == str(max(m.message_id for m in (old_status, undelivered_old, delivered_old, old_logs)))
    assert await _remaining(message_repo) == {str(m.message_id) for m in (young_status, young_logs, mid_logs)}


@pytest.mark.anyio
async def test_delete_expired_boundary(message_repo):
    just_over = _status(timedelta(days=2, seconds=1))
    just_under = _status(timedelta(days=2) - timedelta(seconds=1))
    await message_repo.add_message(just_over)
    await message_repo.add_message(just_under)

    deleted, _ = await message_repo.delete_expired(**RETENTION)

    assert deleted == 1
    assert await _remaining(message_repo) == {str(just_under.message_id)}


@pytest.mark.anyio
async def test_delete_expired_batches_with_keyset(message_repo):
    msgs = [_status(timedelta(days=10, seconds=-i)) for i in range(5)]
    for m in msgs:
        await message_repo.add_message(m)
    ids = sorted(str(m.message_id) for m in msgs)

    deleted, last_id = await message_repo.delete_expired(batch_size=2, **RETENTION)
    assert (deleted, last_id) == (2, ids[1])
    assert await _remaining(message_repo) == set(ids[2:])

    deleted, last_id = await message_repo.delete_expired(batch_size=2, last_message_id=last_id, **RETENTION)
    assert (deleted, last_id) == (2, ids[3])

    deleted, last_id = await message_repo.delete_expired(batch_size=2, last_message_id=last_id, **RETENTION)
    assert (deleted, last_id) == (1, ids[4])
    assert await _remaining(message_repo) == set()

    assert await message_repo.delete_expired(batch_size=2, **RETENTION) == (0, None)


@pytest.mark.anyio
async def test_delete_expired_select_uses_primary_key_range(message_repo):
    for m in (_status(timedelta(days=10)), _logs(timedelta(days=20))):
        await message_repo.add_message(m)
    captured = {}
    original = message_repo.get_rows

    async def spy(sql, params=None):
        captured["sql"], captured["params"] = sql, params
        return await original(sql, params)

    message_repo.get_rows = spy
    await message_repo.delete_expired(last_message_id="0" * 26, **RETENTION)
    plan = await original("EXPLAIN QUERY PLAN " + captured["sql"], captured["params"])
    detail = " ".join(str(r["detail"]) for r in plan)
    assert "SEARCH" in detail and "sqlite_autoindex_messages_1" in detail and "SCAN" not in detail
