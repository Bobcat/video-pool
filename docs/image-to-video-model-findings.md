# Image-To-Video Model Findings

This note records the first I2V model selection for video-pool.

## Goal

Add a real image-to-video model that fits the current pool design:

- usable from the workbench
- compatible with the runtime admin API
- small enough to test without dedicating the full GPU
- served through a backend we already have, if possible

## Current State

The two Wan2.1 models currently configured in video-pool are text-to-video only.

The workbench can show an image input, but those models advertise:

- `tasks: ["text_to_video"]`
- `input_modalities: ["text"]`
- `max_images: 0`

They should not accept image-to-video requests.

## Candidate Tested

The first candidate was the local Wan2.2 TI2V 5B 4-step NVFP4 package.

Local files:

- `/home/gunnar/models/Wan2.2-TI2V-5B`
- `/home/gunnar/models/Wan2.2-TI2V-5B-4step-nvfp4`

The base Wan2.2 TI2V 5B model supports both text-to-video and image-to-video.
The model card describes it as a 720p, 24 FPS model that can run on a single
consumer GPU such as a 4090.

The local 4-step NVFP4 package looked like the best first test because:

- it is already downloaded
- it is a 5B model, not a 14B model
- it is designed for 4-8 step inference
- video-pool already has a `lightx2v_serve` backend

## Result

Do not expose this package as a `lightx2v_serve` model in video-pool yet.

The `auswolf/Wan2.2-TI2V-5B-4step-nvfp4` repo is a ComfyUI package. Its model
card ships a ComfyUI workflow, a TAE file, and one NVFP4 safetensors file. It
does not ship a separate LightX2V checkpoint or calibration file.

The local file fails with LightX2V in two ways:

- With `dit_quant_scheme: "nvfp4"`, startup fails because LightX2V expects
  `input_global_scale` / `alpha` tensors or `calib.pt` data.
- With `dit_quant_scheme: "mxfp4"`, startup succeeds, but inference fails with
  `RuntimeError: Inconsistency of Tensor type:scale_b`.

The local official `Wan-AI/Wan2.2-TI2V-5B` directory is also incomplete for a
direct LightX2V run. It has VAE and T5 files, but the three
`diffusion_pytorch_model-*.safetensors` transformer shards are not present.

## Model Added

The first working I2V model in video-pool is:

- `wan2.1-i2v-14b-480p-int8-lightx2v`

It uses the local Wan2.1 I2V 14B 480p LightX2V directory plus the official
LightX2V INT8 distilled DiT checkpoint:

- `/home/gunnar/models/Wan2.1-I2V-14B-480P-LightX2V`
- `/home/gunnar/models/Wan2.1-Distill-Models/wan2.1_i2v_480p_int8_lightx2v_4step.safetensors`

The INT8 DiT checkpoint must be loaded as a quantized checkpoint:

- `dit_quantized: true`
- `dit_quant_scheme: "int8-triton"`
- `dit_quantized_ckpt: ".../wan2.1_i2v_480p_int8_lightx2v_4step.safetensors"`

Loading the same file as `dit_original_ckpt` fails with a dtype mismatch. The
model expects the LightX2V INT8 path.

The current config also uses quantized T5 and CLIP encoders, lazy load, and
block-level CPU offload. This keeps peak VRAM lower by streaming model blocks
through the GPU instead of keeping the full pipeline resident.

The model is image-to-video, not an image editor. It can animate an input image.
It should not be expected to perform instruction edits such as changing a shoe
brand while preserving text.

## Viable Next Routes

| Route | Notes | Verdict |
| --- | --- | --- |
| ComfyUI backend | Matches the package format and workflow. Requires a new backend or managed ComfyUI server integration. | Best fit for the local 4-step package. |
| Complete official 5B download | Download the missing Wan2.2 TI2V 5B transformer shards and configure LightX2V with the upstream `wan_ti2v_i2v.json` style config. | Useful, but not the same fast 4-step package. |
| LightX2V-compatible quant | Find or create a true LightX2V-compatible 5B quant with the needed NVFP4 scale data. | Good if such an artifact exists. Do not guess scales in production. |

## Other Options

| Option | Notes | First-pass verdict |
| --- | --- | --- |
| Wan2.2 TI2V 5B 4-step NVFP4 | Local, fast path, I2V capable, but the available package is ComfyUI-format. | Needs ComfyUI backend or conversion. |
| LTX Video 0.9.8 / 13B distilled | Interesting fast family with I2V support. Needs separate integration work. | Good second family. |
| CogVideoX I2V | Diffusers-friendly baseline, but likely slower and older. | Later baseline. |
| Stable Video Diffusion XT | Simple I2V baseline, but older and limited. | Only useful as a fallback smoke test. |
| HunyuanVideo I2V | Potentially strong, but heavier and more backend work. | Defer. |
| FramePack / CausVid | Interesting fast research paths. Likely custom backend work. | Defer. |

## Sources

- Fast video model collection: https://huggingface.co/collections/linoyts/fast-video-generation-models
- Wan2.2 TI2V 5B: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Wan2.2 TI2V 5B 4-step NVFP4 package: https://huggingface.co/auswolf/Wan2.2-TI2V-5B-4step-nvfp4
- LTX Video: https://huggingface.co/Lightricks/LTX-Video
- Stable Video Diffusion XT: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- CogVideoX I2V: https://huggingface.co/zai-org/CogVideoX-5b-I2V
