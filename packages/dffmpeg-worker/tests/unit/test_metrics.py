import asyncio

import pytest

from dffmpeg.worker.metrics import (
    ACCEL_HARDWARE,
    ACCEL_NONE,
    ACCEL_SOFTWARE,
    EncoderMetrics,
    MetricsServer,
    _video_encoder_from_args,
    classify_encoder,
)

FFMPEG = "/usr/lib/jellyfin-ffmpeg/ffmpeg"


class TestVideoEncoderFromArgs:
    def test_finds_stream_specific_codec(self):
        assert _video_encoder_from_args(["-i", "in.mkv", "-c:v", "h264_vaapi", "out.ts"]) == "h264_vaapi"

    def test_finds_indexed_stream_specific_codec(self):
        # The form Jellyfin actually emits.
        assert _video_encoder_from_args(["-codec:v:0", "h264_vaapi", "out.ts"]) == "h264_vaapi"

    def test_finds_vcodec_alias(self):
        assert _video_encoder_from_args(["-vcodec", "libx264"]) == "libx264"

    def test_ignores_audio_and_subtitle_codecs(self):
        # The regression this guards: an audio-only transcode must not be
        # counted as a software VIDEO encode.
        assert _video_encoder_from_args(["-c:a", "aac", "-c:s", "copy"]) is None

    def test_video_specific_beats_generic(self):
        assert _video_encoder_from_args(["-c", "copy", "-c:v", "h264_vaapi"]) == "h264_vaapi"
        # Order must not matter -- specificity does.
        assert _video_encoder_from_args(["-c:v", "h264_vaapi", "-c", "copy"]) == "h264_vaapi"

    def test_last_video_specific_wins(self):
        assert _video_encoder_from_args(["-c:v", "libx264", "-c:v", "h264_vaapi"]) == "h264_vaapi"

    def test_generic_codec_used_when_no_video_specific(self):
        assert _video_encoder_from_args(["-c", "copy"]) == "copy"

    def test_no_codec_option_at_all(self):
        assert _video_encoder_from_args(["-i", "in.mkv", "-f", "null", "-"]) is None

    def test_trailing_flag_without_value(self):
        assert _video_encoder_from_args(["-i", "in.mkv", "-c:v"]) is None

    def test_flag_followed_by_another_flag(self):
        assert _video_encoder_from_args(["-c:v", "-hide_banner"]) is None


class TestClassifyEncoder:
    @pytest.mark.parametrize(
        "encoder",
        ["h264_vaapi", "hevc_vaapi", "av1_qsv", "h264_nvenc", "hevc_videotoolbox"],
    )
    def test_hardware_suffixes(self, encoder):
        assert classify_encoder(encoder)[1] == ACCEL_HARDWARE

    @pytest.mark.parametrize("encoder", ["libx264", "libx265", "libsvtav1", "mpeg2video"])
    def test_software(self, encoder):
        assert classify_encoder(encoder)[1] == ACCEL_SOFTWARE

    def test_copy_is_not_an_encode(self):
        assert classify_encoder("copy") == ("copy", ACCEL_NONE)

    def test_unknown_hardware_encoder_still_classifies_as_hardware(self):
        # The load-bearing property: a future Intel/AMD encoder name must not
        # land in the software bucket and fire a false regression alert.
        label, accel = classify_encoder("vvc_vaapi")
        assert accel == ACCEL_HARDWARE
        assert label == "other"

    def test_unknown_name_is_collapsed_to_other(self):
        assert classify_encoder("../../etc/passwd")[0] == "other"

    def test_case_is_normalized(self):
        assert classify_encoder("H264_VAAPI") == ("h264_vaapi", ACCEL_HARDWARE)


class TestEncoderMetrics:
    def test_records_a_completed_hardware_encode(self):
        m = EncoderMetrics()
        m.record_job(FFMPEG, ["-c:v", "h264_vaapi"], 0)
        assert list(m.samples()) == [({"encoder": "h264_vaapi", "accel": ACCEL_HARDWARE}, 1)]

    def test_counts_accumulate(self):
        m = EncoderMetrics()
        for _ in range(3):
            m.record_job(FFMPEG, ["-c:v", "libx264"], 0)
        assert list(m.samples()) == [({"encoder": "libx264", "accel": ACCEL_SOFTWARE}, 3)]

    def test_ignores_failed_jobs(self):
        m = EncoderMetrics()
        m.record_job(FFMPEG, ["-c:v", "libx264"], 1)
        assert list(m.samples()) == []

    def test_ignores_non_ffmpeg_binaries(self):
        m = EncoderMetrics()
        m.record_job("/usr/lib/jellyfin-ffmpeg/ffprobe", ["-c:v", "libx264"], 0)
        assert list(m.samples()) == []

    def test_ignores_jobs_that_encode_no_video(self):
        m = EncoderMetrics()
        m.record_job(FFMPEG, ["-i", "in.mkv", "-c:a", "aac", "out.m4a"], 0)
        assert list(m.samples()) == []

    def test_render_is_valid_exposition_text(self):
        m = EncoderMetrics()
        m.record_job(FFMPEG, ["-c:v", "h264_vaapi"], 0)
        out = m.render()
        assert "# TYPE dffmpeg_worker_jobs_by_encoder_total counter" in out
        assert 'dffmpeg_worker_jobs_by_encoder_total{encoder="h264_vaapi",accel="hardware"} 1' in out
        assert out.endswith("\n")

    def test_render_with_no_samples_still_emits_headers(self):
        out = EncoderMetrics().render()
        assert "# HELP dffmpeg_worker_jobs_by_encoder_total" in out
        assert "\ndffmpeg_worker_jobs_by_encoder_total{" not in out


class TestMetricsServer:
    @pytest.mark.asyncio
    async def test_serves_metrics_and_404s_everything_else(self):
        metrics = EncoderMetrics()
        metrics.record_job(FFMPEG, ["-c:v", "h264_vaapi"], 0)
        server = MetricsServer("127.0.0.1", 0, metrics=metrics)
        await server.start()
        assert server._server is not None
        port = server._server.sockets[0].getsockname()[1]

        async def request(path):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
            await writer.drain()
            data = await reader.read()
            writer.close()
            return data.decode()

        try:
            ok = await request("/metrics")
            assert ok.startswith("HTTP/1.1 200 OK")
            assert 'encoder="h264_vaapi"' in ok

            missing = await request("/healthz")
            assert missing.startswith("HTTP/1.1 404")
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_bind_failure_is_not_fatal(self):
        first = MetricsServer("127.0.0.1", 0)
        await first.start()
        assert first._server is not None
        port = first._server.sockets[0].getsockname()[1]
        try:
            second = MetricsServer("127.0.0.1", port)
            await second.start()
            # Swallowed, not raised: metrics must never stop the worker.
            assert second._server is None
        finally:
            await first.stop()
