#!/usr/bin/env bash
# Convert a verl Megatron checkpoint (SFT or RL) to Hugging Face format, restore frozen modules, verify.
#
#   scripts/export_hf.sh <global_step_dir> <out_dir> [base_model]
#
#   base_model defaults to project.model.base. Pass the smoke / 9B model id for those runs, e.g.
#   scripts/export_hf.sh checkpoints/sft/sft_smoke_0p8b/global_step_5 /tmp/hf Qwen/Qwen3.5-0.8B
#
# 1. verl model merger (Megatron dist-ckpt → HF safetensors). RL checkpoints keep the model under
#    global_step_N/actor/; SFT checkpoints (verl sft_trainer) save it directly in global_step_N/.
# 2. copy the vision tower back from the base checkpoint (exact freeze; AdamW decay drifts it otherwise)
# 3. verify: frozen tensors == base, some language weights changed (catches silent no-op exports)
source "$(dirname "$0")/common.sh"
CKPT="${1:?global_step_N directory}"; OUT="${2:?output HF dir}"
BASE="${3:-$(cfg project.model.base)}"
SRC="${CKPT}"; [[ -d "${CKPT}/actor" ]] && SRC="${CKPT}/actor"
HF=""
for d in "${SRC}/huggingface" "${SRC}/model/huggingface"; do
  [[ -f "${d}/config.json" && -f "${d}/model.safetensors.index.json" ]] && { HF="${d}"; break; }
done
if [[ -n "${HF}" ]]; then
  # Megatron-Bridge checkpoints (engine use_mbridge) already hold the HF safetensors. verl's merger reads only the
  # dist-ckpt, which is empty then, and fails on Qwen3.5's nested config ("no attribute num_hidden_layers", 2026-10-29).
  log "copying the bridge's HF weights ${HF} → ${OUT} (base for frozen modules: ${BASE})"
  mkdir -p "${OUT}" && cp -a "${HF}/." "${OUT}/"
  # the base's tokenizer and processor files that the bridge does not write (merges.txt, vocab.json, preprocessors)
  "${PY}" - "${BASE}" "${OUT}" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

arg, out = sys.argv[1], Path(sys.argv[2])
base = Path(arg) if Path(arg).is_dir() else Path(snapshot_download(arg, local_files_only=True))
for f in sorted(base.iterdir()):
    if f.is_file() and ".safetensors" not in f.name and not (out / f.name).exists():
        shutil.copy2(f, out / f.name)
        print(f"added from {arg}: {f.name}")
PY
else
  log "merging ${SRC} → ${OUT} (base for frozen modules: ${BASE})"
  "${PY}" -m verl.model_merger merge --backend megatron --local_dir "${SRC}" --target_dir "${OUT}"
fi
"${PY}" -m dpt.launch.checkpoint_tools restore-frozen --ckpt "${OUT}" --base "${BASE}" --prefix model.visual.
"${PY}" -m dpt.launch.checkpoint_tools verify --ckpt "${OUT}" --base "${BASE}" --prefix model.visual.
log "exported ${OUT}"
