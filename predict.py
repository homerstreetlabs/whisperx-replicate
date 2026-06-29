from cog import BasePredictor, Input, Path, BaseModel
from typing import Any, Optional
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
from whisperx.diarize import DiarizationPipeline

import atexit
import faulthandler
import gc
import math
import os
import shutil
import signal
import sys
import whisperx
import tempfile
import threading
import time
import torch
import traceback
import ffmpeg
import logging

os.environ.setdefault(
    "COG_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3"
# Diarization model is baked into the image at build time (see build.sh) and loaded from this
# local directory so it does NOT depend on the caller-supplied HF token. pyannote resolves the
# gated `community-1` weights (and its bundled segmentation/embedding/plda sub-models) from local
# disk, so callers whose tokens never accepted the community-1 user agreement still get diarization.
diarization_model_dir = "./models/diarization/speaker-diarization-community-1"
# English alignment model (torchaudio WAV2VEC2_ASR_BASE_960H). Baked into the image and copied
# into torch.hub's checkpoint cache in setup() so whisperx's align step loads it locally instead
# of downloading ~360MB at runtime. Only covers English; other languages still resolve via HF.
align_model_filename = "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
# Adaptive diarization segmentation step. 90% window overlap (0.1) gives the best speaker-boundary
# precision, but its clustering cost scales ~O(N²) with audio length and OOMs / overruns the worker
# health check on very long files. Episodes over the threshold use a coarser step (fewer analysis
# windows → far cheaper clustering, slightly coarser turn boundaries); shorter episodes keep the
# high-quality default.
diarization_long_episode_hours = 5.0
diarization_long_episode_step = 0.33   # > 5h
diarization_default_step = 0.1         # <= 5h (community-1 / pyannote default)

logging.basicConfig(level=logging.INFO)


# --- Temporary memory tracing (diagnosing intermittent diarization-stage worker kills) ---
# An OOM kill (SIGKILL) and a native segfault both die with NO Python traceback, so the
# only way to tell them apart is to sample memory live. The last "[mem] sample" line before
# the worker dies tells us: (a) which sub-step killed it, and (b) whether host RAM / cgroup
# memory was at its limit (OOM) or had plenty of headroom (native crash).

def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _proc_kb(path, key):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def memory_snapshot(include_gpu=True):
    g = 1024 ** 3
    parts = []
    # GPU stats touch the CUDA API. Only query them from the main thread (include_gpu=True);
    # the background sampler passes include_gpu=False so it never races main-thread CUDA
    # context init / model .to(device) — a rare but real native-crash vector.
    if include_gpu:
        try:
            if torch.cuda.is_available():
                parts.append(f"gpu_alloc={torch.cuda.memory_allocated() / g:.2f}G")
                parts.append(f"gpu_reserved={torch.cuda.memory_reserved() / g:.2f}G")
                parts.append(f"gpu_peak={torch.cuda.max_memory_reserved() / g:.2f}G")
        except Exception:
            pass

    rss = _proc_kb("/proc/self/status", "VmRSS:")
    if rss is not None:
        parts.append(f"rss={rss / g:.2f}G")

    # cgroup v2, then v1 fallback — this is the number the OOM killer actually watches.
    cur = _read_int("/sys/fs/cgroup/memory.current")
    mx = _read_int("/sys/fs/cgroup/memory.max")
    if cur is None:
        cur = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        mx = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cur is not None:
        if mx is not None and mx < (1 << 62):
            parts.append(f"cgroup={cur / g:.2f}/{mx / g:.2f}G")
        else:
            parts.append(f"cgroup={cur / g:.2f}G")

    avail = _proc_kb("/proc/meminfo", "MemAvailable:")
    if avail is not None:
        parts.append(f"sys_avail={avail / g:.2f}G")

    return "  ".join(parts) if parts else "n/a"


def log_memory(tag, include_gpu=True):
    try:
        logging.info(f"[mem] {tag}: {memory_snapshot(include_gpu=include_gpu)}")
    except Exception:
        pass


class MemorySampler(threading.Thread):
    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            # Background thread: never touch CUDA APIs here (include_gpu=False). Host
            # RSS/cgroup is what we need for OOM detection; GPU stats are sampled on the
            # main thread at key checkpoints instead.
            log_memory("sample", include_gpu=False)

    def stop(self):
        self._stop.set()


class memory_tracer:
    def __init__(self, interval=5.0):
        self._sampler = MemorySampler(interval)

    def __enter__(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        log_memory("trace start")
        self._sampler.start()
        return self

    def __exit__(self, *exc):
        self._sampler.stop()
        log_memory("trace end")
        return False
# --- end temporary memory tracing ---


# --- Temporary crash diagnostics (catch NON-OOM worker deaths) ---
# An OOM kill arrives as SIGKILL, which is uncatchable — the memory sampler above is how we
# catch that case. EVERY OTHER death mode is catchable and handled here:
#   * native segfault in pyannote/torch (SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL) -> faulthandler
#   * orchestrator termination / timeout (SIGTERM/SIGINT)                       -> signal logger
#   * uncaught exception inside the sampler thread                              -> threading hook
# If the worker dies and NONE of these fire, that itself confirms a hard SIGKILL (OOM).

# Turns a silent native crash into a full all-threads Python traceback on stderr.
faulthandler.enable(all_threads=True)


def _thread_excepthook(args):
    logging.error(
        "[crash] unhandled exception in thread %s",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    log_memory("at thread exception")


threading.excepthook = _thread_excepthook


@atexit.register
def _log_clean_exit():
    # Runs only on an orderly interpreter shutdown. If the worker is hard-killed (SIGKILL from
    # the OOM killer, or a segfault), this line will be ABSENT — its absence is diagnostic.
    logging.info("[crash] process exiting via atexit (orderly shutdown, not a hard kill)")


_TERMINATION_LOGGING_INSTALLED = False


def install_termination_logging():
    # Best-effort logging of catchable termination signals. Chains to whatever handler is
    # already installed (e.g. cog's) so we observe without changing shutdown behavior. Called
    # from setup() so it runs after cog has wired up its own handlers. signal.signal() only
    # works on the main thread, so failures are swallowed.
    global _TERMINATION_LOGGING_INSTALLED
    if _TERMINATION_LOGGING_INSTALLED:
        return
    _TERMINATION_LOGGING_INSTALLED = True

    def _make_handler(signame, prev):
        def _handler(signum, frame):
            logging.error("[crash] received %s (%s) — worker is being terminated", signame, signum)
            log_memory(f"at {signame}")
            try:
                traceback.print_stack(frame)
                sys.stderr.flush()
            except Exception:
                pass
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
        return _handler

    for signame in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            prev = signal.getsignal(signum)
            signal.signal(signum, _make_handler(signame, prev))
        except Exception as e:
            logging.warning("[crash] could not install %s handler: %s", signame, e)
# --- end temporary crash diagnostics ---


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        install_termination_logging()

        destination_folder = '../root/.cache/torch'
        os.makedirs(destination_folder, exist_ok=True)

        # VAD model -> torch cache (legacy; whisperx may already bundle this).
        vad_source = os.path.join('./models/vad', 'whisperx-vad-segmentation.bin')
        if os.path.exists(vad_source):
            vad_dest = os.path.join(destination_folder, 'whisperx-vad-segmentation.bin')
            if not os.path.exists(vad_dest):
                shutil.copy(vad_source, destination_folder)

        # English alignment model -> torch.hub checkpoint cache, so the align step loads it
        # locally instead of downloading ~360MB at runtime.
        align_source = os.path.join('./models/align', align_model_filename)
        if os.path.exists(align_source):
            hub_ckpt_dir = os.path.join(destination_folder, 'hub', 'checkpoints')
            os.makedirs(hub_ckpt_dir, exist_ok=True)
            align_dest = os.path.join(hub_ckpt_dir, align_model_filename)
            if not os.path.exists(align_dest):
                shutil.copy(align_source, align_dest)

        # Load the diarization model ONCE at boot rather than per-prediction. The per-prediction
        # Pipeline.from_pretrained(...).to(device) was dying intermittently on the FIRST prediction
        # of a cold-booted worker (first CUDA init + GPU load, under the prediction health-check
        # window). Doing it here moves that fragile load into the more forgiving boot window, and
        # applies pyannote's in-memory Lightning checkpoint upgrade once here instead of in the
        # prediction path. Best-effort: on failure we fall back to per-prediction loading so the
        # worker still boots (no crash-loop). Only possible when the model is baked in (no caller
        # token needed); otherwise it stays None and loads per-prediction with the caller's token.
        self.diarize_model = None
        if os.path.isdir(diarization_model_dir):
            try:
                log_memory("before diarize preload (setup)")
                self.diarize_model = load_diarization_model(token=None)
                log_memory("diarize model preloaded (setup)")
            except Exception:
                logging.exception("[setup] diarization preload failed; will load per-prediction")

    def run(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: Optional[str] = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            task: str = Input(
                description="Task to perform on the audio file. Options are: transcribe, translate (English only)",
                choices=["transcribe", "translate"],
                default="transcribe"),
            initial_prompt: Optional[str] = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: Optional[str] = Input(
                description="HuggingFace token (read). Optional: the diarization model is baked into the image, so a "
                            "token is only needed as a fallback if the model is not pre-cached. If provided, the "
                            "account must have accepted the pyannote/speaker-diarization-community-1 user agreement.",
                default=None),
            min_speakers: Optional[int] = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: Optional[int] = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            user_agent: Optional[str] = Input(
                description="Override the User-Agent used to download the audio file. Useful when the host "
                            "blocks the default value.",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=True),  # TEMP: default-on to capture per-stage timings while diagnosing diarization crashes
            episode_id: Optional[int] = Input(
                description="Episode ID for webhook correlation",
                default=None),
            user_id: Optional[int] = Input(
                description="User ID for webhook correlation",
                default=None),
    ) -> Output:
        if user_agent:
            os.environ["COG_USER_AGENT"] = user_agent

        with memory_tracer(), torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            audio_duration = get_audio_duration(audio_file)

            if language is None and language_detection_min_prob > 0 and audio_duration > 30000:
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms)
                )

                segments_starts = distribute_segments_equally(audio_duration, segments_duration_ms,
                                                              language_detection_max_tries)

                logging.info("Detecting languages on segments starting at " + ', '.join(map(str, segments_starts)))

                detected_language_details = detect_language(audio_file, segments_starts, language_detection_min_prob,
                                                            language_detection_max_tries, asr_options, vad_options, task)

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                logging.info(f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                      f"{detected_language_iterations} iterations.")

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options=asr_options, vad_options=vad_options, task=task)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                logging.info(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                logging.info(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                logging.info(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                if detected_language in DEFAULT_ALIGN_MODELS_TORCH or detected_language in DEFAULT_ALIGN_MODELS_HF:
                    result = align(audio, result, debug)
                else:
                    logging.info(f"Cannot align output as language {detected_language} is not supported for alignment")

            if diarization:
                result = diarize(self.diarize_model, audio, result, debug, huggingface_access_token, min_speakers, max_speakers)

            if debug:
                logging.info(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)

    stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio stream found in {file_path}")
    if stream and "duration" in stream:
        return float(stream["duration"]) * 1000

    # Fallback to format duration if stream duration is not available
    if "format" in probe and "duration" in probe["format"]:
        return float(probe["format"]["duration"]) * 1000

    raise ValueError("Could not determine audio duration from file metadata")


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, task):
    model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, asr_options=asr_options, task=task,
                                vad_options=vad_options)
    try:
        best = None
        for iteration, start_ms in enumerate(segments_starts, start=1):
            audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)
            try:
                audio = whisperx.load_audio(audio_segment_file_path)
                model_n_mels = model.model.feat_kwargs.get("feature_size")
                segment = log_mel_spectrogram(
                    audio[:N_SAMPLES],
                    n_mels=model_n_mels if model_n_mels is not None else 80,
                    padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0]
                )
                encoder_output = model.model.encode(segment)
                results = model.model.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]
            finally:
                audio_segment_file_path.unlink()

            logging.info(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

            detected = {"language": language, "probability": language_probability, "iterations": iteration}
            if best is None or language_probability > best["probability"]:
                best = detected
            if language_probability >= language_detection_min_prob:
                break

        return best
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        del model


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        logging.info(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg
                .input(input_file_path, ss=start_time_ms/1000)
                .output(temp_file.name, t=duration_ms/1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logging.info("ffmpeg error occurred: ", e.stderr.decode('utf-8'))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logging.info(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def load_diarization_model(token):
    # Build the whisperx DiarizationPipeline, preferring the image-baked local model (no token /
    # network needed) and falling back to a runtime HF download. One-shot retry clears the CUDA
    # allocator and tries again on a *catchable* fault; a native segfault / SIGKILL during the GPU
    # load kills the process outright — which is why setup() preloads this at boot, off the
    # prediction path, for cold-boot-heavy deployments.
    if os.path.isdir(diarization_model_dir):
        model_source = diarization_model_dir
    else:
        model_source = "pyannote/speaker-diarization-community-1"
        if not token:
            raise ValueError(
                f"Diarization model not found at {diarization_model_dir} and no "
                "huggingface_access_token was provided to download it. Run ./build.sh (with "
                "HF_TOKEN set) before `cog push`, or pass a token whose account accepted the "
                "pyannote/speaker-diarization-community-1 user agreement."
            )

    last_err = None
    for attempt in range(1, 3):
        try:
            return DiarizationPipeline(model_name=model_source, token=token, device=device)
        except Exception as e:
            last_err = e
            logging.exception("[diarize] load attempt %d/2 failed for %s", attempt, model_source)
            gc.collect()
            torch.cuda.empty_cache()
            if attempt < 2:
                time.sleep(1.0)
    raise last_err


def _set_segmentation_step(diarize_model, seg_step):
    # whisperx's DiarizationPipeline wraps a pyannote SpeakerDiarization pipeline as `.model`.
    # segmentation_step controls the overlap of the ~10s analysis windows; the underlying Inference
    # reads its `.step` (in seconds) live at inference time, so mutating it after load is safe — the
    # same pattern pyannote uses for `segmentation_batch_size`. Best-effort: on any internal-API
    # mismatch we log and leave the model's default step in place (no regression, just no speedup).
    try:
        pipe = diarize_model.model
        inference = pipe._segmentation
        inference.step = seg_step * inference.duration
        pipe.segmentation_step = seg_step
        logging.info(
            "[diarize] segmentation_step=%.2f applied (window=%.1fs, step=%.2fs)",
            seg_step, inference.duration, inference.step,
        )
        return True
    except Exception as e:
        logging.warning(
            "[diarize] could not apply segmentation_step=%.2f (%s); using model default",
            seg_step, e,
        )
        return False


def diarize(diarize_model, audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    log_memory("before diarize")

    # Adaptive segmentation step: long episodes use a coarser step so diarization clustering
    # (cost ~O(N²) in the number of analysis windows) stays within memory/time limits.
    duration_hours = len(audio) / SAMPLE_RATE / 3600
    seg_step = (
        diarization_long_episode_step
        if duration_hours > diarization_long_episode_hours
        else diarization_default_step
    )
    logging.info("[diarize] audio=%.2fh -> segmentation_step=%.2f", duration_hours, seg_step)

    # Normally preloaded once at boot in setup(). Load now only if that didn't happen — the model
    # isn't baked into the image, or the boot-time preload failed on this worker.
    if diarize_model is None:
        diarize_model = load_diarization_model(huggingface_access_token)

    # The pipeline is reused across predictions, so always (re)apply the step for THIS run; a
    # previous long episode may have left it at the coarser value.
    _set_segmentation_step(diarize_model, seg_step)

    log_memory("diarize model ready, before inference")
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    log_memory("after diarize inference")

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logging.info(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    # Free per-inference intermediates but keep the (resident) model loaded for reuse.
    gc.collect()
    torch.cuda.empty_cache()

    return result
