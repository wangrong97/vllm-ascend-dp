export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1/0 是否打屏
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_HOST_LOG_FILE_NUM=1000
# export ASCEND_LAUNCH_BLOCKING=1 # 强制同步日志

export ASCEND_PROCESS_LOG_PATH=/home/w00608002/plog

# source /usr/local/Ascend/driver/bin/setenv.bash
# source /usr/local/Ascend/cann-9.1.T560/set_env.sh
# source /usr/local/Ascend/cann/set_env.sh

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# vllm serve /mnt/share/weight/dsk_v4-flash-w8a8_mxfp-smooth-0707-full
vllm serve /home/weight/DeepSeek-V4-Flash \
  --max_model_len 135000 \
  --safetensors-load-strategy 'prefetch' \
  --max-num-batched-tokens 4096  \
  --served-model-name dsv \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 64 \
  --port 9000 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --data-parallel-size 4 \
  --api_server_count 1 \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager": true}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/yuanlinfeng/vllm_profiling", "torch_profiler_with_stack": false}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true, "enable_qkv_pseudo_quant": true }' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  # --enforce-eager \
  # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       
