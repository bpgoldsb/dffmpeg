"""
Fire-and-forget telemetry pushes to the in-cluster Pushgateway.

Nothing in this module is allowed to raise, and every push is bounded by
``config.telemetry_timeout`` -- a Pushgateway outage or slow network must
never delay or fail the surrounding transcode ``exec`` (spec JXC-8). Callers
should not wrap these in their own try/except for that reason; the guarantee
is made here, once, so it can't be forgotten at a call site.

Three things are pushed, all deliberately simple text-exposition-format
bodies (no client library dependency needed for a handful of lines):

* ``push_invocation_outcome`` -- a single gauge recording which path the
  *last* invocation took ("dffmpeg", "local_failsafe", or "local_fallback").
  Pushed to a STABLE per-client grouping key via POST (merge, not replace) so
  it coexists with the heartbeat gauge below under the same key. This is what
  makes JXC-9's three states ("never attempted" / "attempted and failed" /
  "succeeded") distinguishable from the exported metrics.
* ``push_fallback_event`` -- increments the fallback counter (JXC-8). Because
  Pushgateway has no server-side increment (a push to the same grouping key
  just replaces that metric's value), each fallback event is pushed under its
  OWN grouping key (client_id + a random suffix) so one fallback is never
  silently overwritten by the next. The tradeoff -- something must eventually
  garbage-collect old per-event series from the gateway -- is a known
  Pushgateway limitation for this pattern and is called out as a follow-up
  for whoever builds the downstream alert/dashboard (not this work item).
* ``push_heartbeat`` / ``heartbeat_loop_task`` -- a periodic timestamp gauge
  (JXC-15), independent of whether any fallback fired, so a downstream
  dead-man's-switch alert can tell a silent/down Pushgateway apart from a
  healthy dffmpeg path with nothing to report.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

import httpx
from dffmpeg.client.config import ClientConfig

logger = logging.getLogger(__name__)

# Metric/job names pushed to Pushgateway. Kept as module constants (not
# config) so the metric *names* stay stable across the fleet even when the
# Pushgateway URL/timeout is overridden per-pod.
_JOB_NAME = "dffmpeg_client"
_FALLBACK_JOB_NAME = "dffmpeg_client_fallback"


def _client_instance_label(config: ClientConfig) -> str:
    return config.client_id or "unknown"


async def _push(config: ClientConfig, job: str, instance: str, body: str, method: str = "POST") -> None:
    """
    Sends one push to the Pushgateway. Swallows every exception (including a
    non-2xx response, which is only logged at debug) -- this function must
    never propagate a failure to its caller.
    """
    if not config.pushgateway_enabled:
        return

    url = f"{config.pushgateway_url.rstrip('/')}/metrics/job/{job}/instance/{instance}"

    try:
        async with httpx.AsyncClient(timeout=config.telemetry_timeout) as http:
            if method == "PUT":
                resp = await http.put(url, content=body)
            else:
                resp = await http.post(url, content=body)
            if resp.status_code >= 300:
                logger.debug(f"Pushgateway push to {url} returned HTTP {resp.status_code}")
    except Exception as e:
        # Telemetry must never break playback (JXC-8). Debug, not warning --
        # a flapping Pushgateway shouldn't spam every transcode's logs.
        logger.debug(f"Pushgateway push to {url} failed: {e}")


async def push_invocation_outcome(config: ClientConfig, path: str, reason: Optional[str] = None) -> None:
    """
    Records which path this invocation took.

    Args:
        path: one of "dffmpeg", "local_failsafe", or "local_fallback".
        reason: for path == "local_fallback", the pre-output failure reason
            (also recorded on the fallback counter itself).
    """
    ts = time.time()
    lines = [
        "# TYPE dffmpeg_client_last_invocation_path gauge",
        f'dffmpeg_client_last_invocation_path{{path="{path}"}} 1',
        "# TYPE dffmpeg_client_last_invocation_timestamp_seconds gauge",
        f"dffmpeg_client_last_invocation_timestamp_seconds {ts}",
    ]
    await _push(config, _JOB_NAME, _client_instance_label(config), "\n".join(lines) + "\n", method="POST")

    if path == "local_fallback":
        await push_fallback_event(config, reason=reason or "unknown")


async def push_fallback_event(config: ClientConfig, reason: str) -> None:
    """Increments the fallback counter for one fallback event (JXC-8)."""
    event_instance = f"{_client_instance_label(config)}.{uuid.uuid4().hex[:12]}"
    lines = [
        "# TYPE dffmpeg_client_fallback_total counter",
        f'dffmpeg_client_fallback_total{{reason="{reason}"}} 1',
    ]
    await _push(config, _FALLBACK_JOB_NAME, event_instance, "\n".join(lines) + "\n", method="PUT")


async def push_heartbeat(config: ClientConfig) -> None:
    """Pushes the periodic dead-man's-switch heartbeat gauge (JXC-15)."""
    ts = time.time()
    lines = [
        "# TYPE dffmpeg_client_heartbeat_timestamp_seconds gauge",
        f"dffmpeg_client_heartbeat_timestamp_seconds {ts}",
    ]
    await _push(config, _JOB_NAME, _client_instance_label(config), "\n".join(lines) + "\n", method="POST")


async def heartbeat_loop_task(config: ClientConfig, interval: float) -> None:
    """
    Pushes a heartbeat on a fixed interval until cancelled. Intended to run as
    a background asyncio.Task alongside a long-running dffmpeg job stream, so
    the JXC-15 heartbeat keeps flowing for multi-hour transcodes, not just
    once at invocation start/end.
    """
    while True:
        try:
            await push_heartbeat(config)
        except asyncio.CancelledError:
            raise
        except Exception:
            # push_heartbeat already swallows its own errors; this is
            # belt-and-braces so a bug here can never surface as a job
            # failure.
            logger.debug("Heartbeat push iteration failed", exc_info=True)

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
