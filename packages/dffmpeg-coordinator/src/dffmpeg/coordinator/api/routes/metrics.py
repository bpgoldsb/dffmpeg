from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from dffmpeg.common.models import JobMetricsResponse, MetricCounts
from dffmpeg.coordinator.api.dependencies import (
    get_auth_repo,
    get_config,
    get_job_repo,
    get_worker_repo,
    verify_metrics_ip,
)
from dffmpeg.coordinator.config import CoordinatorConfig
from dffmpeg.coordinator.db.auth import AuthRepository
from dffmpeg.coordinator.db.jobs import JobRepository
from dffmpeg.coordinator.db.workers import WorkerRepository

router = APIRouter()

# Prometheus text exposition format, version 0.0.4 -- see
# https://github.com/prometheus/docs/blob/main/content/docs/instrumenting/exposition_formats.md
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_WINDOWS = ("current", "last_1m", "last_5m")

# Windows rendered for the success-only (`job.status == "completed"`) family below.
# Deliberately excludes "current": a completed job is by definition no longer
# current (`is_current = not is_terminal`), so a "current" window on a
# completed-only family would be a permanently-zero series -- worse than no
# series at all, since a future `== 0` alert on it would fire forever.
_SUCCESS_WINDOWS = ("last_1m", "last_5m")


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_metric_family(
    name: str,
    help_text: str,
    metric_type: str,
    samples: Iterable[Tuple[Dict[str, str], float]],
) -> list:
    """
    Render one Prometheus metric family (HELP + TYPE header, then one line per sample).

    Args:
        name (str): The metric name, e.g. "dffmpeg_workers_online".
        help_text (str): One-line description emitted as a `# HELP` comment.
        metric_type (str): The Prometheus metric type, e.g. "gauge".
        samples (Iterable[Tuple[Dict[str, str], float]]): (labels, value) pairs. An
            empty labels dict emits a bare `name value` line.

    Returns:
        list[str]: The rendered lines for this metric family, in exposition-text order.
    """
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"]
    for labels, value in samples:
        if labels:
            label_str = ",".join(f'{key}="{_escape_label_value(str(val))}"' for key, val in labels.items())
            lines.append(f"{name}{{{label_str}}} {value}")
        else:
            lines.append(f"{name} {value}")
    return lines


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(verify_metrics_ip)],
)
async def get_metrics(
    job_repo: JobRepository = Depends(get_job_repo),
    auth_repo: AuthRepository = Depends(get_auth_repo),
    worker_repo: WorkerRepository = Depends(get_worker_repo),
    config: CoordinatorConfig = Depends(get_config),
):
    """
    Exposes job-throughput and worker-availability metrics in Prometheus exposition
    format (JXC-10, JXC-12), for scraping by the existing `dffmpeg` ServiceMonitor.

    Args:
        job_repo (JobRepository): Job repository.
        auth_repo (AuthRepository): Auth repository (used to seed the known worker set).
        worker_repo (WorkerRepository): Worker repository (used for the online-worker count).
        config (CoordinatorConfig): Coordinator configuration.

    Returns:
        PlainTextResponse: The metrics, rendered as Prometheus text-format 0.0.4.
    """
    # 1. Seed empty categories, same as the legacy JSON shape did
    binaries = list(config.allowed_binaries)
    all_clients = await auth_repo.list_identities()
    worker_ids = [w.client_id for w in all_clients if w.role == "worker"]

    # Terminal statuses a job can end in -- defined up front (not just before the
    # tally loop) because it also seeds per_status below, the same way `binaries`/
    # `worker_ids` seed per_binary/per_worker.
    terminal_statuses = {"completed", "failed", "canceled"}

    metrics = JobMetricsResponse(
        total=MetricCounts(),
        per_binary={b: MetricCounts() for b in binaries},
        per_worker={w: MetricCounts() for w in worker_ids},
        per_status={s: MetricCounts() for s in terminal_statuses},
        duration_seconds_sum={window: 0.0 for window in _SUCCESS_WINDOWS},
    )

    # 2. Get recent jobs (last 5 minutes = 300 seconds)
    jobs = await job_repo.get_recent_jobs(window_seconds=300)
    now = datetime.now(timezone.utc)
    cutoff_1m = now.timestamp() - 60
    cutoff_5m = now.timestamp() - 300

    # 3. In-memory tally
    for job in jobs:
        is_terminal = job.status in terminal_statuses
        job_time = job.last_update.timestamp() if job.last_update else 0

        # Current
        is_current = not is_terminal

        # Last 1m
        is_1m = is_current or (is_terminal and job_time >= cutoff_1m)

        # Last 5m
        is_5m = is_current or (is_terminal and job_time >= cutoff_5m)

        # Tally totals
        if is_current:
            metrics.total.current += 1
        if is_1m:
            metrics.total.last_1m += 1
        if is_5m:
            metrics.total.last_5m += 1

        # Tally per binary
        b = job.binary_name
        if b not in metrics.per_binary:
            metrics.per_binary[b] = MetricCounts()
        if is_current:
            metrics.per_binary[b].current += 1
        if is_1m:
            metrics.per_binary[b].last_1m += 1
        if is_5m:
            metrics.per_binary[b].last_5m += 1

        # Tally per worker
        w = job.worker_id
        if w:
            if w not in metrics.per_worker:
                metrics.per_worker[w] = MetricCounts()
            if is_current:
                metrics.per_worker[w].current += 1
            if is_1m:
                metrics.per_worker[w].last_1m += 1
            if is_5m:
                metrics.per_worker[w].last_5m += 1

        # Tally per (terminal) status. A non-terminal job (pending/assigned/
        # running/canceling) contributes to none of these keys -- only the three
        # seeded above are ever populated, so `dffmpeg_jobs_by_status` never grows
        # an unbounded label set. `is_current` is included for structural parity
        # with the per_binary/per_worker blocks above, but it is always False
        # here: is_current == not is_terminal, and this branch only runs when
        # job.status IS terminal, so every status="..." ,window="current" sample
        # in the rendered family is permanently 0 -- expected, not a bug.
        if job.status in terminal_statuses:
            s = job.status
            if is_current:
                metrics.per_status[s].current += 1
            if is_1m:
                metrics.per_status[s].last_1m += 1
            if is_5m:
                metrics.per_status[s].last_5m += 1

        # Tally successes (job.status == "completed"), keyed on job.last_update --
        # which, for a job that reached "completed", IS its completion time
        # (update_status writes last_update alongside status on every transition).
        # Keying on last_update (not created_at) matters: get_recent_jobs bounds
        # its candidate set on last_update, so a long-running job that finished
        # seconds ago but was created >5m ago is still in `jobs` above -- bucketing
        # on created_at would make it uncountable here. `current` is never set;
        # see _SUCCESS_WINDOWS.
        if job.status == "completed":
            if job_time >= cutoff_1m:
                metrics.succeeded.last_1m += 1
            if job_time >= cutoff_5m:
                metrics.succeeded.last_5m += 1

            # Per-job duration (last_update - created_at, seconds), summed into
            # the same windows as the success tally above. created_at is
            # non-optional on Job (always set at submission time); last_update
            # is guarded because job_time already treats a missing one as 0,
            # which would otherwise inflate this sum with a bogus negative-ish
            # duration.
            if job.last_update is not None:
                duration_seconds = (job.last_update - job.created_at).total_seconds()
                if job_time >= cutoff_1m:
                    metrics.duration_seconds_sum["last_1m"] += duration_seconds
                if job_time >= cutoff_5m:
                    metrics.duration_seconds_sum["last_5m"] += duration_seconds

    # 4. Online-worker count (JXC-10) -- sourced from worker_repo, NOT from the
    # per_worker job-throughput buckets above, which are identity-seeded and carry
    # no online/offline signal (research R4/R5/R23).
    online_workers = await worker_repo.get_workers_by_status("online")

    lines: list = []

    lines.extend(
        _render_metric_family(
            "dffmpeg_workers_online",
            "Number of dffmpeg workers currently reporting an online status.",
            "gauge",
            [({}, len(online_workers))],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_jobs",
            "Job counts over a trailing time window, across all binaries and workers.",
            "gauge",
            [({"window": window}, getattr(metrics.total, window)) for window in _WINDOWS],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_jobs_succeeded",
            "Count of jobs that reached status=completed, bucketed by the trailing "
            "window in which their last_update (== completion time) falls. No "
            "'current' window is emitted -- a completed job is never current, so "
            "that series would be permanently zero.",
            "gauge",
            [({"window": window}, getattr(metrics.succeeded, window)) for window in _SUCCESS_WINDOWS],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_job_duration_seconds_sum",
            "Sum of per-job duration (last_update - created_at, in seconds) for "
            "jobs that reached status=completed, bucketed by the trailing window "
            "in which their last_update (== completion time) falls. Divide by "
            "the matching dffmpeg_jobs_succeeded{window=...} sample for mean job "
            "duration -- no paired count series is emitted since that one "
            "already exists. No 'current' window, same reasoning as "
            "dffmpeg_jobs_succeeded.",
            "gauge",
            [({"window": window}, metrics.duration_seconds_sum[window]) for window in _SUCCESS_WINDOWS],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_jobs_by_binary",
            "Job counts over a trailing time window, broken down by binary name.",
            "gauge",
            [
                ({"binary": binary, "window": window}, getattr(counts, window))
                for binary, counts in metrics.per_binary.items()
                for window in _WINDOWS
            ],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_jobs_by_worker",
            "Job counts over a trailing time window, broken down by worker client ID. "
            "Zero-valued for every known worker identity regardless of online status "
            "-- see dffmpeg_workers_online for actual availability.",
            "gauge",
            [
                ({"worker": worker, "window": window}, getattr(counts, window))
                for worker, counts in metrics.per_worker.items()
                for window in _WINDOWS
            ],
        )
    )

    lines.extend(
        _render_metric_family(
            "dffmpeg_jobs_by_status",
            "Job counts over a trailing time window, broken down by terminal status "
            "(completed/failed/canceled -- see terminal_statuses). Lets an alert "
            "distinguish a worker whose jobs are being canceled (client-driven, "
            "healthy) from one whose jobs are failing (broken) -- neither is visible "
            "in dffmpeg_jobs_succeeded, which only tracks status=completed. All "
            "three statuses always render, zero-valued if unseen, same as "
            "dffmpeg_jobs_by_binary/_by_worker. The 'current' window is always 0 "
            "for every status here: reaching one of these three IS what makes a job "
            "non-current, so that sample can never be nonzero -- it is rendered "
            "anyway only to keep the window set identical to the other by-X "
            "families above.",
            "gauge",
            [
                ({"status": status, "window": window}, getattr(counts, window))
                for status, counts in metrics.per_status.items()
                for window in _WINDOWS
            ],
        )
    )

    body = "\n".join(lines) + "\n"
    return PlainTextResponse(content=body, media_type=PROMETHEUS_CONTENT_TYPE)
