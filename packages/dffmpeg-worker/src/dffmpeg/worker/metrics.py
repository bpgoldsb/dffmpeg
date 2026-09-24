"""
Worker-side Prometheus metrics: which encoder each completed job actually used.

WHY THIS EXISTS. Downstream monitoring needs to answer "are transcodes really
being offloaded to the GPU, or have they silently fallen back to software?".
The only signal previously available was the host's Intel media-engine
busy_ratio, which is a *shared* engine -- anything else on the box using
hardware decode (an NVR, another container) saturates it, so an inference
built on it is confounded in both directions and can sit permanently unable
to fire. The worker is the one process that holds the real ffmpeg argv, so it
is the only place the question can be answered directly instead of inferred.

WHAT IS EXPORTED. One counter family, incremented once per job that selected
a video encoder, whatever became of it:

    dffmpeg_worker_jobs_by_encoder_total{encoder="h264_vaapi",accel="hardware",outcome="canceled"}

``accel`` is the label alerts should key on -- it is derived from the encoder
name's suffix, so a VAAPI/QSV/NVENC encoder this module has never heard of
still classifies as ``hardware`` rather than being miscounted as software.
``encoder`` is clamped to a known allow-list (else ``other``) so a malformed
or adversarial argv cannot mint unbounded label values.

WHY ``outcome`` EXISTS, AND WHY CANCELED JOBS COUNT. The obvious design --
count only clean exits -- was tried first and is wrong here. Real traffic is
~68 jobs/10h of which ~65 are HLS client-aborts and only ~3 complete: a
counter gated on exit code 0 would discard roughly 95% of the evidence and
leave the downstream alert too starved to fire. That is the same shape as the
confounded-gauge bug this module was written to replace, arrived at from the
other direction, so it is called out here rather than left as a footnote.

A canceled job is strong evidence, not weak: ffmpeg had selected its encoder
and was running when the viewer stopped watching. What this counter actually
measures is *which encoder the job's ffmpeg command line selected*, which is
precisely the thing a HardwareAccelerationType regression changes. Alerts
should read ``outcome!="failed"`` -- the encoder was chosen and ran without
erroring out. ``failed`` is broken out rather than dropped because a VAAPI
path that errors is a different fault with its own coverage
(DffmpegWorkerDispatchedAndFailing), and folding it in would let a
hard-failing hardware encode read as a healthy one.

Deliberately no ``prometheus_client`` dependency and no web framework: the
exposition format is a handful of lines (the coordinator hand-renders it the
same way, see coordinator/api/routes/metrics.py) and the worker's only HTTP
need is one read-only endpoint. Adding aiohttp/prometheus_client to a package
whose entire dependency set is three libraries would cost more than it saves.
"""

import asyncio
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_METRIC_NAME = "dffmpeg_worker_jobs_by_encoder_total"
_METRIC_HELP = (
    "dffmpeg jobs by the video encoder their ffmpeg command line selected. accel is derived "
    "from the encoder name: hardware, software, or none (stream copy). outcome is completed, "
    "canceled (client aborted mid-encode -- the majority of real traffic) or failed."
)

# Matches the ffmpeg options that set the VIDEO encoder, and only those:
#   -c:v  -c:v:0  -codec:v  -codec:V:1  -vcodec
# A bare -c / -codec (all streams) is matched too but treated as a weaker
# signal -- see _video_encoder_from_args. -c:a / -codec:s must NOT match, or
# an audio-only transcode would be miscounted as a software video encode.
_CODEC_OPT = re.compile(r"^-(?:(?:c|codec)(?::[vV](?::\d+)?)?|vcodec)$")
_VIDEO_SPECIFIC_OPT = re.compile(r"^-(?:(?:c|codec):[vV](?::\d+)?|vcodec)$")

# Encoder-name suffixes that mean "a hardware encode engine did the work".
# Suffix matching (not an allow-list) is the load-bearing choice here: a new
# Intel/AMD/NVIDIA encoder name must never silently land in the `software`
# bucket and fire a false regression alert.
_HARDWARE_SUFFIXES = (
    "_vaapi",
    "_qsv",
    "_nvenc",
    "_amf",
    "_mf",
    "_videotoolbox",
    "_v4l2m2m",
    "_vulkan",
    "_rkmpp",
    "_omx",
)

# Bounds the `encoder` label's cardinality. Anything outside this set is
# exported as "other" -- the accel classification above is what alerts use, so
# collapsing an unknown name costs nothing operationally while keeping a
# hostile or malformed argv from minting one series per job.
_KNOWN_ENCODERS = frozenset(
    {
        # hardware
        "h264_vaapi",
        "hevc_vaapi",
        "av1_vaapi",
        "vp9_vaapi",
        "vp8_vaapi",
        "mjpeg_vaapi",
        "h264_qsv",
        "hevc_qsv",
        "av1_qsv",
        "vp9_qsv",
        "mjpeg_qsv",
        "h264_nvenc",
        "hevc_nvenc",
        "av1_nvenc",
        # software
        "libx264",
        "libx265",
        "libsvtav1",
        "libaom-av1",
        "librav1e",
        "libvpx",
        "libvpx-vp9",
        "mpeg2video",
        "mpeg4",
        "mjpeg",
        "wmv2",
        "rawvideo",
        # not an encode at all
        "copy",
    }
)

ACCEL_HARDWARE = "hardware"
ACCEL_SOFTWARE = "software"
ACCEL_NONE = "none"

OUTCOME_COMPLETED = "completed"
OUTCOME_CANCELED = "canceled"
OUTCOME_FAILED = "failed"
OUTCOMES = frozenset({OUTCOME_COMPLETED, OUTCOME_CANCELED, OUTCOME_FAILED})


def classify_encoder(encoder: str) -> Tuple[str, str]:
    """
    Maps a raw ffmpeg video-encoder name onto the exported (encoder, accel) labels.

    Args:
        encoder (str): The encoder name as it appeared on the ffmpeg command
            line, e.g. "h264_vaapi", "libx264", "copy".

    Returns:
        Tuple[str, str]: (encoder label, accel label). The encoder label is the
            input when it is a known name, else "other"; accel is "hardware",
            "software" or "none".
    """
    name = encoder.strip()
    lowered = name.lower()

    if lowered == "copy":
        accel = ACCEL_NONE
    elif lowered.endswith(_HARDWARE_SUFFIXES):
        accel = ACCEL_HARDWARE
    else:
        accel = ACCEL_SOFTWARE

    label = lowered if lowered in _KNOWN_ENCODERS else "other"
    return label, accel


def _video_encoder_from_args(arguments: Sequence[str]) -> Optional[str]:
    """
    Extracts the video encoder an ffmpeg command line selects, if any.

    A stream-specific option (-c:v / -codec:v:0 / -vcodec) always wins over a
    generic -c/-codec, and the LAST such option wins -- ffmpeg itself applies
    them left to right, so a later flag overrides an earlier one. Returns None
    when the command line sets no video codec at all (an audio-only transcode,
    an ffprobe-shaped invocation, a capability query), which is the signal to
    count nothing rather than to guess.

    Args:
        arguments (Sequence[str]): The resolved ffmpeg arguments, as executed.

    Returns:
        Optional[str]: The encoder name, or None if the command line selects none.
    """
    specific: Optional[str] = None
    generic: Optional[str] = None

    for index, token in enumerate(arguments):
        if not _CODEC_OPT.match(token):
            continue
        if index + 1 >= len(arguments):
            # Trailing flag with no value -- ffmpeg would reject it; nothing to record.
            continue
        value = arguments[index + 1]
        if value.startswith("-"):
            continue
        if _VIDEO_SPECIFIC_OPT.match(token):
            specific = value
        else:
            generic = value

    if specific is not None:
        return specific
    return generic


class EncoderMetrics:
    """
    In-process tally of completed jobs by video encoder.

    A plain dict, not a client library: every mutation happens on the worker's
    single asyncio event loop, so no locking is needed, and the exposition
    render is a dozen lines (see the module docstring).
    """

    def __init__(self) -> None:
        # (encoder label, accel label, outcome) -> count
        self._counts: Dict[Tuple[str, str], int] = {}

    def record_job(self, binary_path: str, arguments: Sequence[str], outcome: str) -> None:
        """
        Records one finished job, if it is one this counter is about.

        Ignores anything that is not ffmpeg and anything whose command line
        encodes no video (an audio-only transcode, an ffprobe-shaped
        invocation, a capability query) -- those say nothing about video
        offload. Every other terminal outcome is counted and labeled; see the
        module docstring for why canceled jobs are the majority of the signal
        rather than noise to be filtered out.

        Args:
            binary_path (str): The binary the executor ran, e.g.
                "/usr/lib/jellyfin-ffmpeg/ffmpeg".
            arguments (Sequence[str]): The resolved arguments, as executed.
            outcome (str): One of OUTCOMES -- "completed", "canceled" or
                "failed". An unrecognized value is recorded as "failed", so a
                caller bug degrades toward the bucket alerts ignore rather
                than inventing a label value.
        """
        if not binary_path.rsplit("/", 1)[-1].startswith("ffmpeg"):
            return

        encoder = _video_encoder_from_args(arguments)
        if encoder is None:
            return

        if outcome not in OUTCOMES:
            logger.warning(f"Unrecognized job outcome {outcome!r}, recording as failed")
            outcome = OUTCOME_FAILED

        enc_label, accel = classify_encoder(encoder)
        key = (enc_label, accel, outcome)
        self._counts[key] = self._counts.get(key, 0) + 1
        logger.info(f"Recorded job: encoder={enc_label} accel={accel} outcome={outcome}")

    def samples(self) -> Iterable[Tuple[Dict[str, str], int]]:
        """Returns the current (labels, value) pairs, in a stable order."""
        for (encoder, accel, outcome), value in sorted(self._counts.items()):
            yield {"encoder": encoder, "accel": accel, "outcome": outcome}, value

    def render(self) -> str:
        """Renders the counter family in Prometheus text exposition format 0.0.4."""
        lines: List[str] = [
            f"# HELP {_METRIC_NAME} {_METRIC_HELP}",
            f"# TYPE {_METRIC_NAME} counter",
        ]
        for labels, value in self.samples():
            label_str = ",".join(f'{key}="{val}"' for key, val in labels.items())
            lines.append(f"{_METRIC_NAME}{{{label_str}}} {value}")
        return "\n".join(lines) + "\n"


# Module-level singleton. The executor records into it and the HTTP server
# reads it; threading one instance through JobRunner -> executor construction
# would touch the coordinator-facing job plumbing for no behavioral gain.
METRICS = EncoderMetrics()


class MetricsServer:
    """
    Minimal read-only HTTP server exposing GET /metrics.

    Serves exactly one route and answers everything else 404. It never touches
    job state, holds no credentials and emits no secret material, so it runs
    unauthenticated like any other exporter -- network reachability is the
    boundary (in Kubernetes, a ClusterIP Service scraped by a ServiceMonitor).
    """

    # Bounds a request read so a half-open or hostile connection cannot pin a
    # task forever. Generous for a header block; nothing legitimate is near it.
    _READ_LIMIT = 16384
    _READ_TIMEOUT = 5.0

    def __init__(self, host: str, port: int, metrics: Optional[EncoderMetrics] = None) -> None:
        self.host = host
        self.port = port
        self.metrics = metrics if metrics is not None else METRICS
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        """
        Starts listening. A bind failure is logged and swallowed: metrics are
        observability, and a port collision must not stop the worker from
        transcoding.
        """
        try:
            self._server = await asyncio.start_server(self._handle, self.host, self.port)
            logger.info(f"Metrics server listening on {self.host}:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server on {self.host}:{self.port}: {e}")
            self._server = None

    async def stop(self) -> None:
        """Stops listening and waits for the listener to close."""
        if self._server is None:
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception as e:
            logger.debug(f"Metrics server close raised: {e}")
        self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=self._READ_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Malformed, oversized, truncated or slow -- drop it silently.
            writer.close()
            return

        try:
            request_line = request.split(b"\r\n", 1)[0].decode("latin-1")
            parts = request_line.split(" ")
            method = parts[0] if parts else ""
            path = parts[1].split("?", 1)[0] if len(parts) > 1 else ""

            if method in ("GET", "HEAD") and path == "/metrics":
                body = self.metrics.render().encode("utf-8")
                head = (
                    "HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {PROMETHEUS_CONTENT_TYPE}\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("latin-1")
                writer.write(head + (b"" if method == "HEAD" else body))
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Metrics request handling failed: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass
