# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Environment variables used by FastDeploy.
"""

import os
import sys
from types import ModuleType
from typing import Any, Callable


def _validate_split_kv_size(value: int) -> int:
    """Validate FD_DETERMINISTIC_SPLIT_KV_SIZE is a positive power of 2."""
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"FD_DETERMINISTIC_SPLIT_KV_SIZE must be a positive power of 2, got {value}.")
    return value


environment_variables: dict[str, Callable[[], Any]] = {
    # Whether to use BF16 on CPU.
    "FD_CPU_USE_BF16": lambda: os.getenv("FD_CPU_USE_BF16", "False"),
    # Cuda architecture to build FastDeploy.This is a list of strings
    # such as [80,90].
    "FD_BUILDING_ARCS": lambda: os.getenv("FD_BUILDING_ARCS", "[]"),
    # Log directory.
    "FD_LOG_DIR": lambda: os.getenv("FD_LOG_DIR", "log"),
    # Global log level, prefer this over FD_DEBUG. Supports "INFO" and "DEBUG".
    "FD_LOG_LEVEL": lambda: os.getenv("FD_LOG_LEVEL", None),
    # Whether to use debug mode, can set 0 or 1
    "FD_DEBUG": lambda: int(os.getenv("FD_DEBUG", "0")),
    # Request logging master switch. Set to 0 to disable request logging.
    "FD_LOG_REQUESTS": lambda: int(os.getenv("FD_LOG_REQUESTS", "1")),
    # Request logging detail level (0-3). Higher level means more verbose output.
    "FD_LOG_REQUESTS_LEVEL": lambda: int(os.getenv("FD_LOG_REQUESTS_LEVEL", "0")),
    # Max field length for request logging truncation.
    "FD_LOG_MAX_LEN": lambda: int(os.getenv("FD_LOG_MAX_LEN", "2048")),
    # Unified trace mode: off, local, otel, all.
    "FD_TRACE": lambda: os.getenv("FD_TRACE", "off"),
    # Number of days to keep fastdeploy logs.
    "FD_LOG_BACKUP_COUNT": lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),
    # Model download source, can set "AISTUDIO", "MODELSCOPE" or "HUGGINGFACE".
    "FD_MODEL_SOURCE": lambda: os.getenv("FD_MODEL_SOURCE", "AISTUDIO"),
    # Model download cache directory.
    "FD_MODEL_CACHE": lambda: os.getenv("FD_MODEL_CACHE", None),
    # Maximum number of stop sequences.
    "FD_MAX_STOP_SEQS_NUM": lambda: int(os.getenv("FD_MAX_STOP_SEQS_NUM", "5")),
    # Maximum length of stop sequences.
    "FD_STOP_SEQS_MAX_LEN": lambda: int(os.getenv("FD_STOP_SEQS_MAX_LEN", "8")),
    # GPU devices that will be used. This is a string that
    # splited by comma, such as 0,1,2.
    "CUDA_VISIBLE_DEVICES": lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),
    # Whether to use HuggingFace tokenizer.
    "FD_USE_HF_TOKENIZER": lambda: bool(int(os.getenv("FD_USE_HF_TOKENIZER", "0"))),
    # Set the high watermark (HWM) for receiving data during ZMQ initialization
    "FD_ZMQ_SNDHWM": lambda: os.getenv("FD_ZMQ_SNDHWM", 0),
    # cache kv quant params directory
    "FD_CACHE_PARAMS": lambda: os.getenv("FD_CACHE_PARAMS", "none"),
    # Set attention backend. "NATIVE_ATTN", "APPEND_ATTN"
    # and "MLA_ATTN" can be set currently.
    "FD_ATTENTION_BACKEND": lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),
    # Set sampling class. "base", "base_non_truncated", "air" and "rejection" can be set currently.
    "FD_SAMPLING_CLASS": lambda: os.getenv("FD_SAMPLING_CLASS", "base"),
    # Set moe backend."cutlass","marlin", "triton", "flashinfer-cutlass", "flashinfer-cutedsl" and "flashinfer-trtllm" can be set currently.
    "FD_MOE_BACKEND": lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),
    # Set nvfp4 load interleaved weight scale.
    "FD_NVFP4_LOAD_BLOCKSCALE_LEAVE": lambda: bool(int(os.getenv("FD_NVFP4_LOAD_BLOCKSCALE_LEAVE", "0"))),
    # Set mxfp4 backend."flashinfer" can be set currently.
    "FD_MOE_MXFP4_BACKEND": lambda: os.getenv("FD_MOE_MXFP4_BACKEND", "flashinfer"),
    # Whether to use Machete for wint4 dense gemm.
    "FD_USE_MACHETE": lambda: os.getenv("FD_USE_MACHETE", "1"),
    # Set whether to disable recompute the request when the KV cache is full.
    "FD_DISABLED_RECOVER": lambda: os.getenv("FD_DISABLED_RECOVER", "0"),
    # Set triton kernel JIT compilation directory.
    "FD_TRITON_KERNEL_CACHE_DIR": lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),
    # Whether transition from standalone PD decoupling to centralized inference
    "FD_PD_CHANGEABLE": lambda: os.getenv("FD_PD_CHANGEABLE", "0"),
    # Whether to use DeepGemm for FP8 blockwise MoE.
    "FD_USE_DEEP_GEMM": lambda: bool(int(os.getenv("FD_USE_DEEP_GEMM", "0"))),
    # Whether to use DeepGemm for FP8 blockwise MoE.
    "FD_USE_BLACKWELL_GEMM": lambda: bool(int(os.getenv("FD_USE_BLACKWELL_GEMM", "0"))),
    # Whether to use PFCCLab/DeepEP.
    "FD_USE_PFCC_DEEP_EP": lambda: bool(int(os.getenv("FD_USE_PFCC_DEEP_EP", "0"))),
    # Whether to use aggregate send.
    "FD_USE_AGGREGATE_SEND": lambda: bool(int(os.getenv("FD_USE_AGGREGATE_SEND", "0"))),
    # Whether to open Trace.
    "TRACES_ENABLE": lambda: os.getenv("TRACES_ENABLE", "false"),
    # set traec Server name.
    "FD_SERVICE_NAME": lambda: os.getenv("FD_SERVICE_NAME", "FastDeploy"),
    # set traec host name.
    "FD_HOST_NAME": lambda: os.getenv("FD_HOST_NAME", "localhost"),
    # set traec exporter.
    "TRACES_EXPORTER": lambda: os.getenv("TRACES_EXPORTER", "console"),
    # set traec exporter_otlp_endpoint.
    "EXPORTER_OTLP_ENDPOINT": lambda: os.getenv("EXPORTER_OTLP_ENDPOINT"),
    # set traec exporter_otlp_headers.
    "EXPORTER_OTLP_HEADERS": lambda: os.getenv("EXPORTER_OTLP_HEADERS"),
    # enable kv cache block scheduler v1 (no need for kv_cache_ratio)
    "ENABLE_V1_KVCACHE_SCHEDULER": lambda: int(os.getenv("ENABLE_V1_KVCACHE_SCHEDULER", "1")),
    # set prealloc block num for decoder
    "FD_ENC_DEC_BLOCK_NUM": lambda: int(os.getenv("FD_ENC_DEC_BLOCK_NUM", "2")),
    # enbale max prefill of one execute step
    "FD_ENABLE_MAX_PREFILL": lambda: int(os.getenv("FD_ENABLE_MAX_PREFILL", "0")),
    # Whether to use PLUGINS.
    "FD_PLUGINS": lambda: None if "FD_PLUGINS" not in os.environ else os.environ["FD_PLUGINS"].split(","),
    # set trace attribute job_id.
    "FD_JOB_ID": lambda: os.getenv("FD_JOB_ID"),
    # support max connections
    "FD_SUPPORT_MAX_CONNECTIONS": lambda: int(os.getenv("FD_SUPPORT_MAX_CONNECTIONS", "1024")),
    # Offset for Tensor Parallelism group GID.
    "FD_TP_GROUP_GID_OFFSET": lambda: int(os.getenv("FD_TP_GROUP_GID_OFFSET", "1000")),
    # enable multi api server
    "FD_ENABLE_MULTI_API_SERVER": lambda: bool(int(os.getenv("FD_ENABLE_MULTI_API_SERVER", "0"))),
    "FD_FOR_TORCH_MODEL_FORMAT": lambda: bool(int(os.getenv("FD_FOR_TORCH_MODEL_FORMAT", "0"))),
    # force disable default chunked prefill
    "FD_DISABLE_CHUNKED_PREFILL": lambda: bool(int(os.getenv("FD_DISABLE_CHUNKED_PREFILL", "0"))),
    # Whether to use new get_output and save_output method (0 or 1)
    "FD_USE_GET_SAVE_OUTPUT_V1": lambda: bool(int(os.getenv("FD_USE_GET_SAVE_OUTPUT_V1", "0"))),
    # Whether to enable model cache feature
    "FD_ENABLE_MODEL_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_CACHE", "0"))),
    # Whether to print scheduler prefill/decode batch logs.
    "FD_CONSOLE_SCHEDULER_METRICS": lambda: bool(int(os.getenv("FD_CONSOLE_SCHEDULER_METRICS", "1"))),
    # Decode log interval for scheduler metrics logs.
    "FD_CONSOLE_DECODE_LOG_INTERVAL": lambda: int(os.getenv("FD_CONSOLE_DECODE_LOG_INTERVAL", "5")),
    # enable internal module to access LLMEngine.
    "FD_ENABLE_INTERNAL_ADAPTER": lambda: int(os.getenv("FD_ENABLE_INTERNAL_ADAPTER", "0")),
    # LLMEngine receive requests port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORT": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORT", None),
    # LLMEngine send response port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORT": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORT", None),
    # LLMEngine receive requests port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORTS", None),
    # LLMEngine send response port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", None),
    # LLMEngine receive control command port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_CONTROL_CMD_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_CONTROL_CMD_SERVER_PORTS", "8202"),
    # Whether to enable the decode caches requests for preallocating resource
    "FD_ENABLE_CACHE_TASK": lambda: os.getenv("FD_ENABLE_CACHE_TASK", "0"),
    # Max pre-fetch requests number in PD
    "FD_EP_MAX_PREFETCH_TASK_NUM": lambda: int(os.getenv("FD_EP_MAX_PREFETCH_TASK_NUM", "8")),
    # Enable or disable model caching.
    # When enabled, the quantized model is stored as a cache for future inference to improve loading efficiency.
    "FD_ENABLE_MODEL_LOAD_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_LOAD_CACHE", "0"))),
    # Whether to clear cpu cache when clearing model weights.
    "FD_ENABLE_SWAP_SPACE_CLEARING": lambda: int(os.getenv("FD_ENABLE_SWAP_SPACE_CLEARING", "0")),
    # enable return text, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ENABLE_RETURN_TEXT": lambda: bool(int(os.getenv("FD_ENABLE_RETURN_TEXT", "0"))),
    # Used to truncate the string inserted during thinking when reasoning in a model. (</think> for ernie-45-vl, \n</think>\n\n for ernie-x1)
    "FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR": lambda: os.getenv("FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR", "</think>"),
    # Timeout for cache_transfer_manager process exit
    "FD_CACHE_PROC_EXIT_TIMEOUT": lambda: int(os.getenv("FD_CACHE_PROC_EXIT_TIMEOUT", "600")),
    # FP4 dense GEMM backend, could be flashinfer-cutlass, flashinfer-trtllm, flashinfer-cudnn or None (default is None)
    "FD_NVFP4_GEMM_BACKEND": lambda: os.getenv("FD_NVFP4_MOE_BACKEND", None),
    # Count for cache_transfer_manager process error
    "FD_CACHE_PROC_ERROR_COUNT": lambda: int(os.getenv("FD_CACHE_PROC_ERROR_COUNT", "10")),
    # API_KEY required for service authentication
    "FD_API_KEY": lambda: [] if "FD_API_KEY" not in os.environ else os.environ["FD_API_KEY"].split(","),
    # The AK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_AK": lambda: os.getenv("ENCODE_FEATURE_BOS_AK"),
    # The SK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_SK": lambda: os.getenv("ENCODE_FEATURE_BOS_SK"),
    # The ENDPOINT of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_ENDPOINT": lambda: os.getenv("ENCODE_FEATURE_ENDPOINT"),
    # Whether the Prefill instance continuously requests Decode resources in PD disaggregation
    "PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES": lambda: int(
        os.getenv("PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES", "1")
    ),
    "FD_ENABLE_E2W_TENSOR_CONVERT": lambda: int(os.getenv("FD_ENABLE_E2W_TENSOR_CONVERT", "0")),
    "FD_ENGINE_TASK_QUEUE_WITH_SHM": lambda: int(os.getenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")),
    "FD_FILL_BITMASK_BATCH": lambda: int(os.getenv("FD_FILL_BITMASK_BATCH", "4")),
    "FD_ENABLE_PDL": lambda: int(os.getenv("FD_ENABLE_PDL", "1")),
    "FD_ENABLE_ASYNC_LLM": lambda: int(os.getenv("FD_ENABLE_ASYNC_LLM", "0")),
    # Enable early RDMA connection for PD disaggregation
    "FD_ENABLE_PD_RDMA_EAGER_CONNECT": lambda: bool(int(os.getenv("FD_ENABLE_PD_RDMA_EAGER_CONNECT", "0"))),
    "FD_GUIDANCE_DISABLE_ADDITIONAL": lambda: bool(int(os.getenv("FD_GUIDANCE_DISABLE_ADDITIONAL", "1"))),
    "FD_LLGUIDANCE_LOG_LEVEL": lambda: int(os.getenv("FD_LLGUIDANCE_LOG_LEVEL", "0")),
    # "Number of tokens in the group for Mixture of Experts (MoE) computation processing on HPU"
    "FD_HPU_CHUNK_SIZE": lambda: int(os.getenv("FD_HPU_CHUNK_SIZE", "64")),
    # "Enable FP8 calibration on HPU"
    "FD_HPU_MEASUREMENT_MODE": lambda: os.getenv("FD_HPU_MEASUREMENT_MODE", "0"),
    "FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS": lambda: int(os.getenv("FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS", "30")),
    "FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE": lambda: int(
        os.getenv("FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", "1")
    ),
    # Whether to collect user information
    "DO_NOT_TRACK": lambda: (os.getenv("DO_NOT_TRACK", "0")) == "1",
    # Usage stats server url
    "FD_USAGE_STATS_SERVER": lambda: os.getenv(
        "FD_USAGE_STATS_SERVER", "http://10.169.17.184:8089/fd/report/periodic"
    ),
    # Usage stats source
    "FD_USAGE_SOURCE": lambda: os.getenv("FD_USAGE_SOURCE", "Unknown"),
    # Usage stats config root
    "FD_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv("FD_CONFIG_ROOT", os.path.join(os.path.expanduser("~"), ".config", "fastdeploy"))
    ),
    "FMQ_CONFIG_JSON": lambda: os.getenv("FMQ_CONFIG_JSON", None),
    "FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": lambda: int(os.getenv("FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "500")),
    "FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": lambda: int(os.getenv("FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "64")),
    "FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT": lambda: float(os.getenv("FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT", "120")),
    "FD_XPU_MOE_FFN_QUANT_TYPE_MAP": lambda: os.getenv("FD_XPU_MOE_FFN_QUANT_TYPE_MAP", ""),
    # Whether to enable low latency in mixed scenario
    "FD_XPU_ENABLE_MIXED_EP_MODE": lambda: bool(int(os.getenv("FD_XPU_ENABLE_MIXED_EP_MODE", "0"))),
    # Reserve output blocks for decoding requests when schedule new prefill requests
    "FD_RESERVE_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL": lambda: int(
        os.getenv("FD_RESERVE_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL", "16")
    ),
    "FD_RESERVE_DECAY_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL": lambda: float(
        os.getenv("FD_RESERVE_DECAY_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL", "0.025")
    ),
    "FD_RESERVE_MIN_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL": lambda: int(
        os.getenv("FD_RESERVE_MIN_OUTPUT_BLOCK_NUM_FOR_DECODE_WHEN_SCHEDULE_NEW_PREFILL", "0")
    ),
    # Timeout for worker process health check in seconds
    "FD_WORKER_ALIVE_TIMEOUT": lambda: int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30")),
    # File path for file storage backend
    "FILE_BACKEND_STORAGE_DIR": lambda: str(os.getenv("FILE_BACKEND_STORAGE_DIR", "/tmp/fastdeploy")),
    # Custom all-reduce max buffer size in MB (default 8MB).
    # Increase this to avoid NCCL fallback for large tensors in deterministic mode.
    # E.g. FD_CUSTOM_AR_MAX_SIZE_MB=128 for 128MB.
    "FD_CUSTOM_AR_MAX_SIZE_MB": lambda: int(os.getenv("FD_CUSTOM_AR_MAX_SIZE_MB", "8")),
    # Enable deterministic inference mode for chunked prefill alignment
    "FD_DETERMINISTIC_MODE": lambda: bool(int(os.getenv("FD_DETERMINISTIC_MODE", "0"))),
    # Split KV block size for deterministic alignment (must be power of 2 and > 0, default 16)
    "FD_DETERMINISTIC_SPLIT_KV_SIZE": lambda: _validate_split_kv_size(
        int(os.getenv("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16"))
    ),
    # Enable determinism logging (print MD5 hashes and debug info)
    "FD_DETERMINISTIC_LOG_MODE": lambda: bool(int(os.getenv("FD_DETERMINISTIC_LOG_MODE", "0"))),
    # Whether to use PD REORDER, can set 0 or 1
    "FD_PD_REORDER": lambda: int(os.getenv("FD_PD_REORDER", "0")),
    # Whether to probe MoE routing probabilities and use Fleet's fused SwiGLU kernel.
    "FD_MOE_PROB_IN_ADVANCE": lambda: bool(int(os.getenv("FD_MOE_PROB_IN_ADVANCE", "0"))),
    # Whether to use batch send data in zmq
    "ZMQ_SEND_BATCH_DATA": lambda: int(os.getenv("ZMQ_SEND_BATCH_DATA", "1")),
    # Whether to enable v1 weight updating, which utilizes ZMQ/EngineWorkerQueue/EngineCacheQueue/FMQs
    # to pass control requests and responses.
    # When v1 is enabled, the legacy /clear_load_weight and /update_model_weight
    # will adopt this new communication pattern.
    "FD_ENABLE_V1_UPDATE_WEIGHTS": lambda: bool(int(os.getenv("FD_ENABLE_V1_UPDATE_WEIGHTS", "0"))),
    # Whether to save the cache of output token for preempted request to storage.
    "FD_SAVE_OUTPUT_CACHE_FOR_PREEMPTED_REQUEST": lambda: bool(
        int(os.getenv("FD_SAVE_OUTPUT_CACHE_FOR_PREEMPTED_REQUEST", "1"))
    ),
    # train-infer consistency, used in RL
    # Whether to align RoPE and moe gate precision with training
    "FD_ENABLE_RL": lambda: int(os.getenv("FD_ENABLE_RL", "0")),
    # Whether to use phi FP8 quantization,if 1,use paddle default.
    "FD_USE_PHI_FP8_QUANT": lambda: bool(int(os.getenv("FD_USE_PHI_FP8_QUANT", "1"))),
    # Enables the Paddle/phi combined TopK operator only when topk_method == noaux_tc,
    # intended for training alignment. Defaults to 0 (disabled).
    "FD_USE_PHI_MOE_TOPK": lambda: bool(int(os.getenv("FD_USE_PHI_MOE_TOPK", "0"))),
    # Whether to use phi MOE permute,if 1,use paddle op.
    "FD_USE_PHI_MOE_PERMUTE": lambda: bool(int(os.getenv("FD_USE_PHI_MOE_PERMUTE", "0"))),
    # Whether to use phi rms_norm,if 1,use paddle op.
    "FD_USE_PHI_RMSNORM": lambda: bool(int(os.getenv("FD_USE_PHI_RMSNORM", "0"))),
    # Control class SiluAndMul to use swiglu or fusid_bias_act operator in the forward_cuda function
    "FD_SiluAndMul_USE_PHI_SWIGLU": lambda: bool(int(os.getenv("FD_SiluAndMul_USE_PHI_SWIGLU", "0"))),
    # Whether to enable FP8 quantization with pow2scale.
    "FD_FP8_QUANT_WITH_POW2SCALE": lambda: bool(int(os.getenv("FD_FP8_QUANT_WITH_POW2SCALE", "0"))),

    # ============================================================
    # Model Download & Source
    # ============================================================
    # Model download source selector (huggingface, aistudio, etc.)
    "DOWNLOAD_SOURCE": lambda: os.getenv("DOWNLOAD_SOURCE", "huggingface"),
    # AIStudio download endpoint URL
    "AISTUDIO_ENDPOINT": lambda: os.getenv("AISTUDIO_ENDPOINT", "http://git.aistudio.baidu.com"),

    # ============================================================
    # Device Selection
    # ============================================================
    # XPU (Kunlun) visible device IDs
    "XPU_VISIBLE_DEVICES": lambda: os.getenv("XPU_VISIBLE_DEVICES", None),
    # Habana HPU visible device IDs
    "HPU_VISIBLE_DEVICES": lambda: os.getenv("HPU_VISIBLE_DEVICES", None),
    # MetaX MACA visible device IDs
    "MACA_VISIBLE_DEVICES": lambda: os.getenv("MACA_VISIBLE_DEVICES", None),
    # MetaX custom device root directory
    "CUSTOM_DEVICE_ROOT": lambda: os.getenv("CUSTOM_DEVICE_ROOT", None),

    # ============================================================
    # PaddlePaddle / SOT Flags
    # ============================================================
    # PaddlePaddle SOT log level
    "SOT_LOG_LEVEL": lambda: os.getenv("SOT_LOG_LEVEL", "0"),
    # SOT unsafe cache fast path
    "SOT_UNSAFE_CACHE_FASTPATH": lambda: os.getenv("SOT_UNSAFE_CACHE_FASTPATH", "1"),
    # SOT zero-size fallback behavior
    "SOT_ENABLE_0_SIZE_FALLBACK": lambda: os.getenv("SOT_ENABLE_0_SIZE_FALLBACK", "0"),
    # SOT specialized dimension numbers
    "SOT_SPECIALIZED_DIM_NUMBERS": lambda: os.getenv("SOT_SPECIALIZED_DIM_NUMBERS", "no"),
    # SOT compilation time limit
    "SOT_ENABLE_COMPILE_TIME_LIMIT": lambda: os.getenv("SOT_ENABLE_COMPILE_TIME_LIMIT", "0"),
    # Specialize device during dynamic-to-static conversion
    "FLAGS_specialize_device_in_dy2st": lambda: os.getenv("FLAGS_specialize_device_in_dy2st", "1"),
    # Enable async fast garbage collection
    "FLAGS_enable_async_fast_gc": lambda: os.getenv("FLAGS_enable_async_fast_gc", "0"),
    # PIR interpreter stream recording for GC cache
    "FLAGS_pir_interpreter_record_stream_for_gc_cache": lambda: os.getenv(
        "FLAGS_pir_interpreter_record_stream_for_gc_cache", "0"
    ),
    # Persistent parameter mode in dynamic-to-static conversion
    "FLAGS_parameters_persistent_mode_in_dy2st": lambda: os.getenv("FLAGS_parameters_persistent_mode_in_dy2st", "1"),
    # Maximum partition size for attention computation kernels
    "FLAGS_max_partition_size": lambda: int(os.getenv("FLAGS_max_partition_size", "1024")),
    # PaddlePaddle selected GPU index for current process
    "FLAGS_selected_gpus": lambda: os.getenv("FLAGS_selected_gpus", "0"),
    # Enable prefill-decode disaggregation mode
    "FLAGS_use_pd_disaggregation": lambda: int(os.getenv("FLAGS_use_pd_disaggregation", "0")),
    # Enable per-chunk prefill-decode disaggregation mode
    "FLAGS_use_pd_disaggregation_per_chunk": lambda: int(os.getenv("FLAGS_use_pd_disaggregation_per_chunk", "0")),
    # Architecture selection for weight-only INT8 linear layers
    "FLAGS_weight_only_linear_arch": lambda: os.getenv("FLAGS_weight_only_linear_arch", None),
    # Flash attention version selector (2 or 3)
    "FLAGS_flash_attn_version": lambda: os.getenv("FLAGS_flash_attn_version", "2"),

    # ============================================================
    # Scheduling & Runtime
    # ============================================================
    # Maximum number of prefill batches during scheduling
    "MAX_PREFILL_NUM": lambda: os.getenv("MAX_PREFILL_NUM", "3"),
    # Controls divided streaming for responses in the router
    "DIVIDED_STREAM": lambda: os.getenv("DIVIDED_STREAM", "0"),
    # Controls whether worker subprocess stdout is unbuffered
    "UNCACHE_WORKER_STDOUT": lambda: os.getenv("UNCACHE_WORKER_STDOUT", "0"),
    # Maximum number of ZMQ dealer connections for engine communication
    "FD_DEALER_CONNECTIONS": lambda: int(os.getenv("FD_DEALER_CONNECTIONS", "50")),
    # Enable objgraph memory debugging for request processing
    "FD_ENABLE_OBJGRAPH_DEBUG": lambda: os.getenv("FD_ENABLE_OBJGRAPH_DEBUG", None),
    # Prefill node one-step stop signal in PD disaggregation
    "PREFILL_NODE_ONE_STEP_STOP": lambda: os.getenv("PREFILL_NODE_ONE_STEP_STOP", "0"),
    # ZMQ message queue port ID for engine-worker communication
    "INFERENCE_MSG_QUEUE_ID": lambda: os.getenv("INFERENCE_MSG_QUEUE_ID", "0"),

    # ============================================================
    # Model Config Overrides
    # ============================================================
    # Model compression ratio, overridable via env var
    "COMPRESSION_RATIO": lambda: os.getenv("COMPRESSION_RATIO", None),
    # RoPE (Rotary Position Embedding) theta parameter
    "ROPE_THETA": lambda: os.getenv("ROPE_THETA", None),

    # ============================================================
    # Speculative Decoding
    # ============================================================
    # Legacy: enable greedy (top-k) verify strategy for speculative decoding
    "SPECULATE_VERIFY_USE_TOPK": lambda: os.getenv("SPECULATE_VERIFY_USE_TOPK", "0"),
    # Legacy: enable target-match verify strategy for speculative decoding
    "SPECULATE_VERIFY_USE_TARGET_SAMPLING": lambda: os.getenv("SPECULATE_VERIFY_USE_TARGET_SAMPLING", "0"),

    # ============================================================
    # TBO & Attention
    # ============================================================
    # Enable Two-Batch Overlap (TBO) optimization
    "USE_TBO": lambda: os.getenv("USE_TBO", "0"),
    # Enable Flash MLA (Multi-head Latent Attention) kernel
    "USE_FLASH_MLA": lambda: os.getenv("USE_FLASH_MLA", "0"),

    # ============================================================
    # HPU (Habana) Hardware
    # ============================================================
    # Enable HPU bucket warmup to pre-compile graph shapes
    "HPU_WARMUP_BUCKET": lambda: int(os.getenv("HPU_WARMUP_BUCKET", "0")),
    # Maximum model sequence length for HPU warmup
    "HPU_WARMUP_MODEL_LEN": lambda: int(os.getenv("HPU_WARMUP_MODEL_LEN", "4096")),
    # HPU warmup: step size for prefill batch sizes
    "BATCH_STEP_PREFILL": lambda: int(os.getenv("BATCH_STEP_PREFILL", "1")),
    # HPU warmup: step size for decode batch sizes
    "BATCH_STEP_DECODE": lambda: int(os.getenv("BATCH_STEP_DECODE", "4")),
    # HPU warmup: step size for decode block numbers
    "BLOCK_STEP_DECODE": lambda: int(os.getenv("BLOCK_STEP_DECODE", "16")),
    # HPU warmup: step size for prefill context block sizes
    "CONTEXT_BLOCK_STEP_PREFILL": lambda: int(os.getenv("CONTEXT_BLOCK_STEP_PREFILL", "1")),
    # HPU warmup: step size for prefill sequence lengths
    "SEQUENCE_STEP_PREFILL": lambda: int(os.getenv("SEQUENCE_STEP_PREFILL", "128")),
    # Enable Habana profiler on HPU hardware
    "HABANA_PROFILE": lambda: int(os.getenv("HABANA_PROFILE", "0")),
    # Sync mode for HPU performance breakdown profiling
    "HPU_PERF_BREAKDOWN_SYNC_MODE": lambda: int(os.getenv("HPU_PERF_BREAKDOWN_SYNC_MODE", "1")),
    # Starting step number for HPU profiling
    "PROFILE_START": lambda: int(os.getenv("PROFILE_START", "0")),
    # Ending step number for HPU profiling
    "PROFILE_END": lambda: int(os.getenv("PROFILE_END", "4")),

    # ============================================================
    # Other Hardware (Iluvatar / MetaX / GCU / XPU)
    # ============================================================
    # Iluvatar KV cache memory allocation in GB
    "FD_ILUVATAR_KVCACHE_MEM": lambda: os.getenv("FD_ILUVATAR_KVCACHE_MEM", "3"),
    # MetaX KV cache memory allocation
    "FD_METAX_KVCACHE_MEM": lambda: os.getenv("FD_METAX_KVCACHE_MEM", None),
    # Enable attention monitoring on GCU (Enflame) hardware
    "FD_GCU_ATTN_MONITOR": lambda: bool(int(os.getenv("FD_GCU_ATTN_MONITOR", "0"))),
    # Process group size for MoE quantization on GCU backend
    "FD_MOE_QUANT_MULTI_PROCESS_GROUP_SIZE": lambda: int(os.getenv("FD_MOE_QUANT_MULTI_PROCESS_GROUP_SIZE", "8")),
    # XPU fused MoE weight-only INT8 TGEMM kernel type selection
    "XFT_MOE_FC_WINT8_TGEMM": lambda: os.getenv("XFT_MOE_FC_WINT8_TGEMM", ""),

    # ============================================================
    # Mooncake Distributed Transfer
    # ============================================================
    # Path to Mooncake configuration file
    "MOONCAKE_CONFIG_PATH": lambda: os.getenv("MOONCAKE_CONFIG_PATH", None),
    # Local hostname for Mooncake transfer engine
    "MOONCAKE_LOCAL_HOSTNAME": lambda: os.getenv("MOONCAKE_LOCAL_HOSTNAME", None),
    # Metadata server address for Mooncake transfer
    "MOONCAKE_METADATA_SERVER": lambda: os.getenv("MOONCAKE_METADATA_SERVER", None),
    # Global segment size for Mooncake transfer
    "MOONCAKE_GLOBAL_SEGMENT_SIZE": lambda: os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None),
    # Local buffer size for Mooncake transfer
    "MOONCAKE_LOCAL_BUFFER_SIZE": lambda: os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", None),
    # Transport protocol for Mooncake transfer (e.g., rdma)
    "MOONCAKE_PROTOCOL": lambda: os.getenv("MOONCAKE_PROTOCOL", "rdma"),
    # RDMA device names for Mooncake transfer
    "MOONCAKE_RDMA_DEVICES": lambda: os.getenv("MOONCAKE_RDMA_DEVICES", ""),
    # Master server address for Mooncake transfer
    "MOONCAKE_MASTER_SERVER_ADDR": lambda: os.getenv("MOONCAKE_MASTER_SERVER_ADDR", None),

    # ============================================================
    # Benchmark & Debug
    # ============================================================
    # Enable saving benchmark results in PyTorch format
    "SAVE_TO_PYTORCH_BENCHMARK_FORMAT": lambda: os.getenv("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", None),
    # Directory for caching compiled Triton kernels
    "TRITON_KERNEL_CACHE_DIR": lambda: os.getenv("TRITON_KERNEL_CACHE_DIR", None),
}


def get_unique_name(self, name):
    """
    Get unique name for config
    """
    shm_uuid = os.getenv("SHM_UUID", "")
    return name + f"_{shm_uuid}"


class _EnvsModule(ModuleType):
    """Custom module class to support __setattr__ for environment variables."""

    def __getattr__(self, name: str):
        if name in environment_variables:
            return environment_variables[name]()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any):
        if name in environment_variables:
            # Convert bool to "1"/"0" so int(os.getenv(...)) works correctly
            if isinstance(value, bool):
                value = int(value)
            os.environ[name] = str(value)
        elif name.startswith("_"):
            # Allow Python-internal attrs (__spec__, __loader__, etc.)
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __delattr__(self, name: str):
        # Support unittest.mock.patch cleanup which calls delattr to restore original state
        if name in environment_variables:
            os.environ.pop(name, None)
        elif name.startswith("_"):
            super().__delattr__(name)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__(self):
        return list(environment_variables.keys())


# Replace the module with our custom class
_current_module = sys.modules[__name__]
_current_module.__class__ = _EnvsModule
