import argparse
import asyncio
import logging
import os
import sys
from typing import Any, AsyncIterator, List, Optional, Tuple, Union

import httpx
from dffmpeg.client.api import DFFmpegClient
from dffmpeg.client.config import ClientConfig, load_config
from dffmpeg.client.telemetry import heartbeat_loop_task, push_invocation_outcome
from dffmpeg.common.cli_utils import (
    add_config_arg,
    add_job_id_arg,
    add_job_subcommand,
    add_window_arg,
    add_worker_subcommand,
    setup_subcommand,
)
from dffmpeg.common.colors import Colors, colorize
from dffmpeg.common.formatting import (
    print_job_details,
    print_job_list,
    print_worker_details,
    print_worker_list,
)
from dffmpeg.common.models import JobLogsMessage, JobRecord, JobStatusMessage
from dffmpeg.common.paths import map_arguments, map_path
from dffmpeg.common.version import get_package_version

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# The pinned, positively-identified fail-fast response shape the coordinator
# returns for a job submission when zero workers are online (spec JXC-13,
# interface contract with the concurrent coordinator work item). Matched
# specifically -- not folded into the generic non-2xx/unreachable case -- so
# it can be told apart in logs/telemetry, per JXC-5's "no worker online" as
# its own named pre-output failure category.
NO_WORKERS_ONLINE_STATUS = 503
NO_WORKERS_ONLINE_ERROR_CODE = "no_workers_online"


class PreOutputFailure(Exception):
    """
    Raised internally by the proxy (dffmpeg_proxy) path to signal that a
    dffmpeg job failed before producing any output, and should fall back to
    the local ffmpeg binary for this invocation (spec JXC-5). `reason` is a
    short machine-readable tag used for logging and fallback telemetry.

    Never raised by the `dffmpeg-client` CLI's own submit/status/etc.
    subcommands -- those keep their existing plain-exit-code behavior.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _is_no_workers_online(exc: httpx.HTTPStatusError) -> bool:
    """
    Positively matches the coordinator's pinned fail-fast response for "zero
    workers online" (HTTP 503, body `{"error": "no_workers_online", ...}`).
    """
    resp = exc.response
    if resp.status_code != NO_WORKERS_ONLINE_STATUS:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error") == NO_WORKERS_ONLINE_ERROR_CODE


async def stream_and_wait(
    client: DFFmpegClient,
    job_id: str,
    transport: str,
    metadata: dict,
    first_message: Optional[Union[JobLogsMessage, JobStatusMessage]] = None,
    agen: Optional[AsyncIterator[Union[JobLogsMessage, JobStatusMessage]]] = None,
) -> int:
    """
    Streams logs and status for a job, waiting for completion.
    Returns exit code (0 for success, 1 for failure/cancellation).

    `first_message` / `agen`: used by the dffmpeg_proxy fallback path, which
    already peeked the first message off `client.stream_job(...)` (bounded by
    the coordinator request timeout, spec JXC-14) before deciding to stream
    normally. When given, that message is replayed and the SAME generator
    object is resumed, rather than opening a second one.
    """
    exit_code = 1

    if agen is None:
        agen = client.stream_job(job_id, transport, metadata)

    async def _iter_with_first() -> AsyncIterator[Union[JobLogsMessage, JobStatusMessage]]:
        if first_message is not None:
            yield first_message
        async for m in agen:  # type: ignore[union-attr]
            yield m

    try:
        async for message in _iter_with_first():
            if isinstance(message, JobLogsMessage):
                for log in message.payload.logs:
                    stream = sys.stdout if log.stream == "stdout" else sys.stderr
                    print(log.content, file=stream)
                    stream.flush()

            elif isinstance(message, JobStatusMessage):
                status = message.payload.status
                if status == "completed":
                    exit_code = message.payload.exit_code if message.payload.exit_code is not None else 0
                    break
                elif status == "failed":
                    exit_code = message.payload.exit_code if message.payload.exit_code is not None else 1
                    break
                elif status == "canceled":
                    exit_code = 130  # Standard SIGINT exit code
                    break

    except asyncio.CancelledError:
        print(colorize("\nCanceling job...", Colors.YELLOW), file=sys.stderr)
        try:
            await client.cancel_job(job_id)
        except Exception as e:
            print(colorize(f"Failed to cancel job: {e}", Colors.RED), file=sys.stderr)
        exit_code = 130
        raise

    return exit_code


def _prepare_job_submission(client: DFFmpegClient, job_args: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Shared argument/path-mapping prep used by both `job_submit` (the `submit`
    CLI subcommand) and the dffmpeg_proxy fallback-aware path below.
    """
    processed_job_args, paths = map_arguments(job_args, client.config.paths)

    cwd = os.getcwd()
    mapped_cwd, used_cwd_var = map_path(cwd, client.config.paths)
    if used_cwd_var and used_cwd_var not in paths:
        paths.append(used_cwd_var)

    return processed_job_args, paths, mapped_cwd


async def _submit_and_await_first_signal(
    client: DFFmpegClient,
    binary_name: str,
    job_args: List[str],
    coordinator_request_timeout: float,
) -> Tuple[JobRecord, Optional[Union[JobLogsMessage, JobStatusMessage]], AsyncIterator[Any]]:
    """
    Submits a job and waits -- bounded by `coordinator_request_timeout` -- for
    the coordinator/worker to show any sign of life (the first status/log
    message on the job's stream). Everything in here that fails is
    classified as a pre-output failure (spec JXC-5) and raised as
    `PreOutputFailure`, maximally tolerant of anything unexpected: an
    unrecognized non-2xx coordinator response, a connection failure, or the
    timeout elapsing with no response at all (spec JXC-14 -- this is what
    catches a worker that's registered `online` but wedged, since `online`
    only reflects a heartbeat, not execution liveness).

    Returns (job, first_message, agen) on success -- `first_message` may be
    None if the stream ended with zero messages (also unusual enough to be
    worth surfacing, but here it's simply passed through so the caller can
    keep streaming normally).
    """
    try:
        processed_job_args, paths, mapped_cwd = _prepare_job_submission(client, job_args)

        try:
            job = await asyncio.wait_for(
                client.submit_job(
                    binary_name,
                    processed_job_args,
                    paths,
                    working_directory=mapped_cwd,
                    monitor=True,
                    heartbeat_interval=client.config.job_heartbeat_interval,
                ),
                timeout=coordinator_request_timeout,
            )
        except asyncio.TimeoutError:
            raise PreOutputFailure("coordinator_timeout_on_submit") from None
        except httpx.HTTPStatusError as e:
            if _is_no_workers_online(e):
                raise PreOutputFailure(NO_WORKERS_ONLINE_ERROR_CODE) from e
            # Tolerant of any other/unrecognized non-2xx shape -- still a
            # positively-classified pre-output failure, never left unhandled.
            raise PreOutputFailure(f"coordinator_http_error_{e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise PreOutputFailure(f"coordinator_unreachable:{type(e).__name__}") from e

        await client._start_heartbeat_loop(str(job.job_id), job.heartbeat_interval)

        agen = client.stream_job(str(job.job_id), job.transport, job.transport_metadata)
        try:
            first_message = await asyncio.wait_for(agen.__anext__(), timeout=coordinator_request_timeout)
        except asyncio.TimeoutError:
            # Coordinator accepted the job (possibly even assigned a worker
            # that reported `online`) but nothing at all came back -- the
            # "wedged worker" case this timeout exists for (JXC-14).
            raise PreOutputFailure("coordinator_timeout_no_execution_signal") from None
        except StopAsyncIteration:
            # Stream ended immediately with no messages at all -- treat the
            # same as "never showed any sign of life".
            raise PreOutputFailure("no_execution_signal") from None

        return job, first_message, agen

    except PreOutputFailure:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Maximally tolerant: anything unanticipated before we've seen a
        # single message from the job is still, by definition, pre-output.
        raise PreOutputFailure(f"unexpected_error:{type(e).__name__}") from e


async def _run_dffmpeg_job(client: DFFmpegClient, config: ClientConfig, binary_name: str, job_args: List[str]) -> int:
    """
    Runs one dffmpeg job end-to-end for the proxy path: submit, wait (bounded)
    for the first sign of execution, then stream to completion while pushing
    a periodic heartbeat (JXC-15) in the background. Raises `PreOutputFailure`
    if the job never gets past the pre-output stage (JXC-5); any failure
    after that point is a normal (non-fallback, mid-stream -- out of scope
    per JXC-6) exit code, same as it always was.
    """
    job, first_message, agen = await _submit_and_await_first_signal(
        client, binary_name, job_args, config.coordinator_request_timeout
    )

    hb_task = asyncio.create_task(heartbeat_loop_task(config, config.telemetry_heartbeat_interval))
    try:
        return await stream_and_wait(
            client,
            str(job.job_id),
            job.transport,
            job.transport_metadata,
            first_message=first_message,
            agen=agen,
        )
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass


def _push_telemetry_sync(config: ClientConfig, path: str, reason: Optional[str] = None) -> None:
    """
    Runs a telemetry push to completion from sync code (proxy_main's tail,
    right before exec/sys.exit -- no event loop is running at that point).
    Bounded by config.telemetry_timeout inside push_invocation_outcome/_push;
    the extra try/except here is defense-in-depth so telemetry genuinely can
    never block or fail the exec (spec JXC-8), no matter what.
    """
    try:
        asyncio.run(push_invocation_outcome(config, path=path, reason=reason))
    except Exception:
        logger.debug("Telemetry push failed", exc_info=True)


def _exec_local(config: ClientConfig, job_args: List[str]) -> None:
    """
    Replaces the current process image with the local jellyfin-ffmpeg binary,
    passing argv through unchanged -- mirrors the calling wrapper script's own
    exec-only error-handling model (no trap, no retry): whatever exit code
    the local binary returns becomes this process's exit code. Never returns
    on success.
    """
    local_path = config.local_ffmpeg_path
    try:
        os.execv(local_path, [local_path] + list(job_args))
    except OSError as e:
        logger.error(f"Failed to exec local ffmpeg at {local_path}: {e}")
        sys.exit(1)


async def job_submit(client: DFFmpegClient, args: argparse.Namespace) -> int:
    # Strip '--' if present in arguments
    job_args = args.arguments
    if job_args and job_args[0] == "--":
        job_args = job_args[1:]

    monitor = not args.detach
    heartbeat_interval = args.heartbeat_interval or client.config.job_heartbeat_interval

    # Process arguments to handle path mapping
    # Note: client.config is accessible
    processed_job_args, paths = map_arguments(job_args, client.config.paths)

    cwd = os.getcwd()
    mapped_cwd, used_cwd_var = map_path(cwd, client.config.paths)
    if used_cwd_var and used_cwd_var not in paths:
        paths.append(used_cwd_var)

    try:
        job = await client.submit_job(
            args.binary,
            processed_job_args,
            paths,
            working_directory=mapped_cwd,
            monitor=monitor,
            heartbeat_interval=heartbeat_interval,
        )

        if not monitor:
            print(colorize("Job submitted successfully.", Colors.GREEN))
            print(f"Job ID: {colorize(str(job.job_id), Colors.CYAN)}")
            return 0

        # Wait/Monitor mode
        await client._start_heartbeat_loop(str(job.job_id), job.heartbeat_interval)
        return await stream_and_wait(client, str(job.job_id), job.transport, job.transport_metadata)

    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        return 1


async def worker_list(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        workers = await client.list_workers(window=args.window)
        print_worker_list(workers)
        return 0
    except Exception as e:
        logger.error(f"Error getting worker list: {e}")
        return 1


async def worker_show(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        worker = await client.get_worker(args.worker_id)
        print_worker_details(worker)
        return 0
    except Exception as e:
        logger.error(f"Error getting worker details: {e}")
        return 1


async def job_list(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        jobs = await client.list_jobs(window=args.window)
        print_job_list(jobs)
        return 0
    except Exception as e:
        logger.error(f"Error getting job list: {e}")
        return 1


async def job_show(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        status = await client.get_job_status(args.job_id)
        print_job_details(status)
        return 0
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        return 1


async def status_cmd(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        print(colorize("=== Workers ===", Colors.MAGENTA))
        await worker_list(client, args)
        print()
        print(colorize("=== Recent Jobs ===", Colors.MAGENTA))
        await job_list(client, args)
        return 0
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return 1


async def job_attach(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        await client.start_monitoring(args.job_id, monitor=True)
        job = await client.get_job_status(args.job_id)
        if job.status in ["completed", "failed", "canceled"]:
            print(f"Job {args.job_id} already finished.")
            return 0
        return await stream_and_wait(client, args.job_id, job.transport, job.transport_metadata)
    except Exception as e:
        logger.error(f"Error attaching to job: {e}")
        return 1


async def job_cancel(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        await client.cancel_job(args.job_id)
        print(f"Job {colorize(args.job_id, Colors.CYAN)} cancellation requested.")
        return 0
    except Exception as e:
        logger.error(f"Error canceling job: {e}")
        return 1


async def job_logs(client: DFFmpegClient, args: argparse.Namespace) -> int:
    try:
        last_msg_id = None
        while True:
            resp = await client.get_job_logs(args.job_id, since_message_id=str(last_msg_id) if last_msg_id else None)
            logs = sorted(resp.logs, key=lambda log: (log.timestamp if log.timestamp else 0))

            for log in logs:
                stream = sys.stdout if log.stream == "stdout" else sys.stderr
                print(log.content, file=stream)
                stream.flush()

            if resp.last_message_id:
                last_msg_id = resp.last_message_id

            if not args.follow:
                if not resp.logs:
                    break
                continue

            job = await client.get_job_status(args.job_id)
            if job.status in ["completed", "failed", "canceled"]:
                break

            if not resp.logs:
                await asyncio.sleep(2)

        # Final check
        resp = await client.get_job_logs(args.job_id, since_message_id=str(last_msg_id) if last_msg_id else None)
        logs = sorted(resp.logs, key=lambda log: (log.timestamp if log.timestamp else 0))
        for log in logs:
            stream = sys.stdout if log.stream == "stdout" else sys.stderr
            print(log.content, file=stream)
            stream.flush()

        return 0
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="dffmpeg client CLI")
    add_config_arg(parser)
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # Submit
    submit_parser = setup_subcommand(subparsers, "submit", "Submit a job", func=job_submit)
    submit_parser.add_argument("--binary", "-b", default="ffmpeg", help="Binary name (default: ffmpeg)")
    submit_parser.add_argument(
        "--detach", "-D", action="store_true", help="Submit job in background and exit immediately"
    )
    submit_parser.add_argument(
        "--heartbeat-interval", type=int, help="Override heartbeat interval (seconds) for this job"
    )
    submit_parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments for the binary")

    # Status
    status_parser = setup_subcommand(subparsers, "status", "Get cluster status", func=status_cmd)
    add_window_arg(status_parser)

    # Worker
    add_worker_subcommand(subparsers, list_func=worker_list, show_func=worker_show)

    # Job
    job_subparsers = add_job_subcommand(
        subparsers,
        list_func=job_list,
        show_func=job_show,
        logs_func=job_logs,
        include_logs_follow=True,
    )

    j_cancel = setup_subcommand(job_subparsers, "cancel", "Cancel a job", func=job_cancel)
    add_job_id_arg(j_cancel)

    j_attach = setup_subcommand(job_subparsers, "attach", "Attach to an existing job", func=job_attach)
    add_job_id_arg(j_attach)

    args = parser.parse_args()

    if args.version:
        print(f"dffmpeg-client {get_package_version('dffmpeg-client')}")
        sys.exit(0)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    async def run():
        async with DFFmpegClient(config) as client:
            if hasattr(args, "func"):
                return await args.func(client, args)
            return 0

    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(130)


def proxy_main():
    """
    Entry point for proxy scripts (e.g. 'ffmpeg').

    Decides local-vs-dffmpeg *before attempting anything* (spec JXC-1/JXC-2):
    dffmpeg is the default path; the `failsafe_force_local` key in the same
    dffmpeg-client.yaml this proxy already re-reads on every invocation (see
    config.load_config / R2) forces local, unconditionally, with no dffmpeg
    attempt at all. This logic lives here rather than in the calling wrapper
    script on purpose -- that script has zero error handling and every branch
    ends in a bare `exec`.

    On the dffmpeg path, a pre-output failure (coordinator unreachable, no
    worker online, or the coordinator request timing out -- spec JXC-5,
    JXC-13, JXC-14) falls back to the local binary for this same invocation,
    after firing a fire-and-forget telemetry push that can never block or
    fail the exec (spec JXC-8). A failure that happens after the job has
    already shown signs of execution (mid-stream) is out of scope (JXC-6) and
    is returned as a normal exit code, exactly as before this change.
    """
    binary_name: str = os.path.basename(sys.argv[0])
    job_args = sys.argv[1:]

    try:
        config = load_config(None)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    if config.failsafe_force_local:
        logger.warning("dffmpeg failsafe (failsafe_force_local) is set; routing directly to local ffmpeg")
        _push_telemetry_sync(config, path="local_failsafe")
        _exec_local(config, job_args)
        return  # pragma: no cover - _exec_local never returns on success

    async def run() -> Tuple[str, Any]:
        async with DFFmpegClient(config) as client:
            try:
                exit_code = await _run_dffmpeg_job(client, config, binary_name, job_args)
                return "ok", exit_code
            except PreOutputFailure as e:
                return "fallback", e.reason
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Anything past this point already saw at least one message
                # from the job (i.e. it's mid-stream, JXC-6 non-goal) -- no
                # fallback, just the same plain-failure behavior every other
                # subcommand here has always had.
                logger.error(f"Unhandled error during dffmpeg job: {e}")
                return "ok", 1

    try:
        outcome, payload = asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(130)

    if outcome == "fallback":
        logger.warning(f"dffmpeg pre-output failure ({payload}); falling back to local ffmpeg")
        _push_telemetry_sync(config, path="local_fallback", reason=payload)
        _exec_local(config, job_args)
        return  # pragma: no cover - _exec_local never returns on success

    _push_telemetry_sync(config, path="dffmpeg")
    sys.exit(payload)


if __name__ == "__main__":
    main()
