#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
      echo "$destination_path already exists. No need to download."
  fi
}

faster_whisper_model_dir=models/faster-whisper-large-v3
mkdir -p $faster_whisper_model_dir

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_model_dir/vocabulary.json"

# --- Alignment model (public): torchaudio WAV2VEC2_ASR_BASE_960H (English) ---
# Baked in so the align step loads locally instead of downloading ~360MB at runtime.
# setup() copies this into torch.hub's checkpoint cache.
align_model_dir=models/align
mkdir -p "$align_model_dir"
download "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth" "$align_model_dir/wav2vec2_fairseq_base_ls960_asr_ls960.pth"

# --- Diarization model (gated): pyannote/speaker-diarization-community-1 ---
# Baked into the image so runtime diarization does NOT depend on the caller-supplied HF token.
# Requires a token whose HF account accepted the user agreement at
# https://hf.co/pyannote/speaker-diarization-community-1
hf_token="${HF_TOKEN:-}"
if [ -z "$hf_token" ] && [ -f hg_access_token.txt ]; then
  hf_token="$(tr -d '[:space:]' < hg_access_token.txt)"
fi
if [ -z "$hf_token" ]; then
  echo "ERROR: HF_TOKEN is required to download the gated diarization model." >&2
  echo "       Set HF_TOKEN=hf_... (account must have accepted" >&2
  echo "       https://hf.co/pyannote/speaker-diarization-community-1 )" >&2
  exit 1
fi

download_gated() {
  local repo_path="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget --header="Authorization: Bearer $hf_token" -O "$destination_path" \
      "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/$repo_path"
  else
    echo "$destination_path already exists. No need to download."
  fi
}

diarization_model_dir=models/diarization/speaker-diarization-community-1
mkdir -p "$diarization_model_dir/segmentation" "$diarization_model_dir/embedding" "$diarization_model_dir/plda"

download_gated "config.yaml"                    "$diarization_model_dir/config.yaml"
download_gated "segmentation/pytorch_model.bin" "$diarization_model_dir/segmentation/pytorch_model.bin"
download_gated "embedding/pytorch_model.bin"    "$diarization_model_dir/embedding/pytorch_model.bin"
download_gated "plda/plda.npz"                  "$diarization_model_dir/plda/plda.npz"
download_gated "plda/xvec_transform.npz"        "$diarization_model_dir/plda/xvec_transform.npz"
