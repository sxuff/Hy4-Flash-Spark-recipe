#!/usr/bin/env bash
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
pin=2cbca389f515df455412dafa07e2b1359cc72a6d
rev=ecd803c859a36bdf9f3614dbd7c548165df74c30
patch_sha=6ec50722c8b181222a928e0a66cda0159e8dff3e72f8193334f6296f6213dfff
cuda=/usr/local/cuda-13.0/bin/nvcc
src="$W/llama.cpp"; bld="$W/build-cuda"; patch="$W/hy4-flash-spark.patch"
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDACXX="$cuda"
mkdir -p "$src"
if [[ ! -d "$src/.git" ]]; then git -C "$src" init; git -C "$src" remote add origin https://github.com/ggml-org/llama.cpp.git; fi
if ! git -C "$src" cat-file -e "$pin^{commit}" 2>/dev/null; then
  for try in 1 2 3; do timeout 120 git -C "$src" fetch --depth 1 origin "$pin" && break; sleep 5; done
fi
[[ $(git -C "$src" rev-parse "$pin^{commit}") = "$pin" ]]
if [[ ! -f "$patch" ]]; then curl -fLsS --retry 4 --max-time 60 "https://huggingface.co/AngelSlim/Hy4-flash-GGUF/raw/$rev/patch/hy4-flash-spark.patch" -o "$patch.part"; mv "$patch.part" "$patch"; fi
printf '%s  %s\n' "$patch_sha" "$patch" | sha256sum -c -
if ! git -C "$src" rev-parse HEAD >/dev/null 2>&1; then git -C "$src" checkout --detach "$pin"; fi
[[ $(git -C "$src" rev-parse HEAD) = "$pin" ]]
if git -C "$src" apply --reverse --check "$patch"; then :; else git -C "$src" apply --check "$patch"; git -C "$src" apply "$patch"; fi
if [[ ${1-} = --source-only ]]; then echo SOURCE_READY; exit; fi
cmake -S "$src" -B "$bld" -G Ninja -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$cuda" -DCMAKE_CUDA_ARCHITECTURES=121a -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$bld" --config Release --target llama-server llama-bench --parallel "${HY4_BUILD_JOBS:-1}"
"$bld/bin/llama-server" --help > "$W/llama-help.txt"
"$bld/bin/llama-server" --list-devices > "$W/devices.txt"
sha256sum "$bld/bin/llama-server" "$bld/bin/llama-bench" > "$W/bin.sha256"
python3 -c 'import json,time; print(json.dumps(dict(status="complete",updated_at=time.time(),upstream="2cbca389f515df455412dafa07e2b1359cc72a6d")))' > "$W/build-status.tmp"
mv "$W/build-status.tmp" "$W/build-status.json"
