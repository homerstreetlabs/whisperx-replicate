from cog import BasePredictor, Input, Path, BaseModel
from typing import Any, Optional
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
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


def memory_snapshot():
    g = 1024 ** 3
    parts = []
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


def log_memory(tag):
    try:
        logging.info(f"[mem] {tag}: {memory_snapshot()}")
    except Exception:
        pass


class MemorySampler(threading.Thread):
    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            log_memory("sample")

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

        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

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
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
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
        if diarization and not huggingface_access_token:
            raise ValueError("huggingface_access_token is required when diarization is enabled")

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
                result = diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)

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


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    log_memory("before diarize model load")
    diarize_model = DiarizationPipeline(token=huggingface_access_token, device=device)
    log_memory("diarize model loaded, before inference")
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    log_memory("after diarize inference")

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logging.info(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
