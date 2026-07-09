# Runtime Admin API

This note documents the video-pool runtime API for inspecting models, loading
and unloading models, and running video generation requests.

The goal is similar to the llm-pool and image-pool runtime admin APIs: routine
model management should not require editing `local.json` and restarting the
service.

It is intentionally a v1 design:

- live runtime control only
- no automatic writes back to `settings.json` or `local.json`
- no arbitrary model definitions via API
- no background job system for model loads
- no force unload or graceful drain state yet

Current reality note:

- this admin API is implemented and is the live control plane used by the
  workbench
- model definitions come from merged `settings.json + local.json`
- load/unload only changes in-process runtime state
- `generation_parameters` and `image_to_video_parameters` are surfaced by public
  and admin model endpoints
- `load_constraints` and `load_override` are surfaced by the admin model endpoint
- load overrides are runtime-only and are passed in the load request body

## Core Concepts

### Configured Model Definition

A configured model definition comes from merged `settings.json + local.json`.

This is static process input. It includes fields such as:

- `backend`
- `enabled`
- `target_inflight`
- `model_path`
- `modalities`
- `output_modalities`
- `tasks`
- `max_images`
- `max_output_videos`
- `vram_estimate_mib`
- `recommended_steps`
- `recommended_guidance`
- `generation_parameters`
- `image_to_video_parameters`

This definition is not modified by the admin API in v1.

### Runtime State

Each configured model also has live runtime state inside the process.

Current states are represented by booleans:

- `loaded`
- `loading`

The workbench maps those booleans to:

- `unloaded`
- `loading`
- `loaded`
- `failed`

Failed load attempts store `last_error`. The model remains unloaded and may be
loaded again.

### Request Parameters

`generation_parameters` is used for text-to-video requests.

`image_to_video_parameters` is used for image-to-video requests.

Each entry is keyed by parameter name. Each entry has a `target`:

- `request`: the value is sent as a top-level request field
- `metadata`: the value is sent inside `metadata`

Entries may include:

- `kind`
- `label`
- `default`
- `minimum`
- `maximum`
- `step`
- `allowed_values`

Supported kinds currently used by video-pool:

- `enum`
- `integer`
- `integer_or_null`
- `number`
- `number_or_null`
- `string`
- `string_or_null`
- `boolean`

### Load Constraints

`load_constraints` describes runtime load controls for a model. These controls
affect model startup or backend server startup, not a single generation request.

The workbench should:

- show these controls on the model details view
- allow editing only when the model is not loaded
- send edited values in `POST /v1/admin/models/{model_name}/load`
- treat `load_override` as the currently active runtime-only override

The admin API does not write load overrides back to config.

## Endpoints

### `GET /v1/models`

Returns loaded models only. This is the public model list for video requests.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "wan2.1-t2v-1.3b-nvfp4-lightx2v",
      "object": "model",
      "owned_by": "video-pool",
      "backend": "lightx2v_serve",
      "capabilities": {
        "input_modalities": ["text"],
        "output_modalities": ["video"],
        "tasks": ["text_to_video"],
        "max_images": 0,
        "max_output_videos": 1
      },
      "recommended_steps": 4,
      "recommended_guidance": 1.0,
      "generation_parameters": {
        "size": {
          "kind": "enum",
          "target": "request",
          "default": "832x480",
          "allowed_values": ["832x480", "480x832"]
        },
        "steps": {
          "kind": "integer",
          "target": "metadata",
          "default": 4,
          "minimum": 1,
          "maximum": 80,
          "step": 1
        }
      },
      "image_to_video_parameters": {}
    }
  ]
}
```

### `GET /v1/admin/models`

Returns all known models from merged config together with live runtime state.
This endpoint is the main UI source of truth for model management.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "wan2.1-t2v-1.3b-nvfp4-lightx2v",
      "backend": "lightx2v_serve",
      "enabled": false,
      "loaded": false,
      "loading": false,
      "loaded_at": null,
      "last_error": null,
      "scheduler": {
        "target_inflight": 1,
        "inflight": 0,
        "queued": 0
      },
      "capabilities": {
        "input_modalities": ["text"],
        "output_modalities": ["video"],
        "tasks": ["text_to_video"],
        "max_images": 0,
        "max_output_videos": 1
      },
      "model_path": "/home/gunnar/models/Wan2.1-T2V-1.3B",
      "vram_estimate_mib": 1500,
      "vram_estimate_source": "configured",
      "recommended_steps": 4,
      "recommended_guidance": 1.0,
      "generation_parameters": {},
      "image_to_video_parameters": {},
      "load_constraints": {
        "lightx2v_cpu_offload": {
          "kind": "boolean",
          "label": "cpu_offload",
          "default": true
        }
      },
      "load_recommendations": {},
      "load_override": {},
      "definition": {
        "model_path": "/home/gunnar/models/Wan2.1-T2V-1.3B",
        "backend": "lightx2v_serve",
        "enabled": false,
        "target_inflight": 1
      }
    }
  ]
}
```

### `GET /v1/admin/gpu-memory`

Returns GPU memory data plus a model-oriented projection for the workbench model
table.

Response shape:

```json
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA ...",
      "used_mib": 68200,
      "total_mib": 98304,
      "free_mib": 30104,
      "used_over_total": "68200MiB / 98304MiB"
    }
  ],
  "models": [
    {
      "name": "wan2.1-t2v-1.3b-nvfp4-lightx2v",
      "backend": "lightx2v_serve",
      "loaded": false,
      "loading": false,
      "vram_estimate_mib": 1500,
      "vram_estimate_source": "configured"
    }
  ],
  "error": null
}
```

### `POST /v1/admin/models/{model_name}/load`

Loads a configured model.

The request body is optional. When present, it may contain fields advertised in
`load_constraints` for that model.

Example:

```json
{
  "lightx2v_cpu_offload": true,
  "lightx2v_t5_cpu_offload": true,
  "lightx2v_sample_shift": 5.0
}
```

Rules:

- unknown fields are rejected
- fields not supported by the model backend are rejected
- overrides are rejected when the model is already loaded
- successful overrides are returned as `load_override`
- overrides are cleared on unload

### `POST /v1/admin/models/{model_name}/unload`

Unloads a loaded model and releases its runtime resources.

There is no separate draining state yet.

### `POST /v1/videos/generations`

Runs text-to-video generation on a loaded model.

Common request fields:

- `model`
- `prompt`
- `n`
- `size`
- `duration_seconds`
- `fps`
- `num_frames`
- `quality`
- `seed`
- `metadata`

Current metadata keys used by backends:

- `steps`
- `guidance`
- `guidance_2`
- `negative_prompt`
- `max_sequence_length`
- `use_prompt_enhancer`
- `lora_name`
- `lora_strength`

### `POST /v1/videos/image-to-video`

Runs image-to-video generation on a loaded model that advertises
`image_to_video`.

The request shape matches text-to-video and adds:

- `images`: input image data URLs

## Backend Notes

### `diffusers_wan_t2v`

Per-request metadata currently used:

- `steps` -> `num_inference_steps`
- `guidance` -> `guidance_scale`
- `guidance_2` -> `guidance_scale_2`
- `negative_prompt`
- `max_sequence_length`

Load overrides currently used:

- `wan_transformer_dtype`
- `wan_vae_dtype`
- `wan_sequential_cpu_offload`
- `wan_vae_tiling`

### `lightx2v_serve`

Per-request metadata currently used:

- `steps` -> `infer_steps`
- `negative_prompt`
- `use_prompt_enhancer`
- `lora_name`
- `lora_strength`

Load overrides are merged into an effective LightX2V JSON config before the
managed LightX2V server process starts.

Load overrides currently used:

- `lightx2v_text_len`
- `lightx2v_sample_guide_scale`
- `lightx2v_sample_shift`
- `lightx2v_enable_cfg`
- `lightx2v_denoising_step_list`
- `lightx2v_cpu_offload`
- `lightx2v_offload_granularity`
- `lightx2v_t5_cpu_offload`
- `lightx2v_vae_cpu_offload`
- `lightx2v_self_attn_1_type`
- `lightx2v_cross_attn_1_type`
- `lightx2v_cross_attn_2_type`
- `lightx2v_rope_type`

## Errors

Errors are returned as JSON objects with an `error` object where possible.

Common error types:

- `unknown_model`
- `model_not_loaded`
- `unsupported_backend`
- `bad_request`
