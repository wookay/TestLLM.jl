module test_testllm_tokugh_ollama_api

using Test
using TestLLM: State, chat1

state = State()
state.system = "あなたはとても粗野なアシスタントです。"
state.model = "mistral"
@info state

chat1
# chat1(state, "こんにちは")


#=
### export OLLAMA_LLM_LIBRARY="cpu_avx2"
### ollama serve
###
time=2025-12-03T07:36:11.506+09:00 level=INFO source=routes.go:1544 msg="server config" env="map[HTTPS_PROXY: HTTP_PROXY: NO_PROXY: OLLAMA_CONTEXT_LENGTH:4096 OLLAMA_DEBUG:INFO OLLAMA_FLASH_ATTENTION:false OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://127.0.0.1:11434 OLLAMA_KEEP_ALIVE:5m0s OLLAMA_KV_CACHE_TYPE: OLLAMA_LLM_LIBRARY:cpu_avx2 OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MODELS:$HOME/.ollama/models OLLAMA_MULTIUSER_CACHE:false OLLAMA_NEW_ENGINE:false OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:1 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://* vscode-file://*] OLLAMA_REMOTES:[ollama.com] OLLAMA_SCHED_SPREAD:false http_proxy: https_proxy: no_proxy:]"
time=2025-12-03T07:36:11.516+09:00 level=INFO source=images.go:522 msg="total blobs: 15"
time=2025-12-03T07:36:11.516+09:00 level=INFO source=images.go:529 msg="total unused blobs removed: 0"
time=2025-12-03T07:36:11.518+09:00 level=INFO source=routes.go:1597 msg="Listening on 127.0.0.1:11434 (version 0.13.0)"
time=2025-12-03T07:36:11.520+09:00 level=INFO source=runner.go:67 msg="discovering available GPUs..."
time=2025-12-03T07:36:11.524+09:00 level=INFO source=server.go:392 msg="starting runner" cmd="/Applications/Ollama.app/Contents/Resources/ollama runner --ollama-engine --port 54434"
time=2025-12-03T07:36:11.628+09:00 level=INFO source=types.go:60 msg="inference compute" id=cpu library=cpu compute="" name=cpu description=cpu libdirs=ollama driver="" pci_id="" type="" total="16.0 GiB" available="8.0 GiB"
time=2025-12-03T07:36:11.629+09:00 level=INFO source=routes.go:1638 msg="entering low vram mode" "total vram"="0 B" threshold="20.0 GiB"
=#

#=
llama_model_loader: loaded meta data with 25 key-value pairs and 291 tensors from $HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Mistral-7B-Instruct-v0.3
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 32768
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 32768
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,32768]   = ["<unk>", "<s>", "</s>", "[INST]", "[...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,32768]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,32768]   = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  20:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 4.07 GiB (4.83 BPW)
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 2 ('</s>')
load: special tokens cache size = 771
load: token to piece cache size = 0.1731 MB
print_info: arch             = llama
print_info: vocab_only       = 1
print_info: model type       = ?B
print_info: model params     = 7.25 B
print_info: general.name     = Mistral-7B-Instruct-v0.3
print_info: vocab type       = SPM
print_info: n_vocab          = 32768
print_info: n_merges         = 0
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: LF token         = 781 '<0x0A>'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 48
llama_model_load: vocab only - skipping tensors
time=2025-12-03T07:36:49.424+09:00 level=INFO source=server.go:392 msg="starting runner" cmd="/Applications/Ollama.app/Contents/Resources/ollama runner --model $HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f --port 54441"
time=2025-12-03T07:36:49.426+09:00 level=INFO source=sched.go:443 msg="system memory" total="16.0 GiB" free="7.7 GiB" free_swap="0 B"
time=2025-12-03T07:36:49.426+09:00 level=INFO source=server.go:459 msg="loading model" "model layers"=33 requested=-1
time=2025-12-03T07:36:49.427+09:00 level=INFO source=device.go:245 msg="model weights" device=CPU size="4.0 GiB"
time=2025-12-03T07:36:49.427+09:00 level=INFO source=device.go:256 msg="kv cache" device=CPU size="512.0 MiB"
time=2025-12-03T07:36:49.427+09:00 level=INFO source=device.go:272 msg="total memory" size="4.5 GiB"
time=2025-12-03T07:36:49.475+09:00 level=INFO source=runner.go:963 msg="starting go runner"
operator() double registration of ggml_uncaught_exception
operator() double registration of ggml_uncaught_exception
operator() double registration of ggml_uncaught_exception
load_backend: loaded CPU backend from /Applications/Ollama.app/Contents/Resources/libggml-cpu-haswell.so
time=2025-12-03T07:36:49.489+09:00 level=INFO source=ggml.go:104 msg=system CPU.0.SSE3=1 CPU.0.SSSE3=1 CPU.0.AVX=1 CPU.0.AVX2=1 CPU.0.F16C=1 CPU.0.FMA=1 CPU.0.BMI2=1 CPU.0.LLAMAFILE=1 CPU.1.SSE3=1 CPU.1.SSSE3=1 CPU.1.LLAMAFILE=1 compiler=cgo(clang)
time=2025-12-03T07:36:49.489+09:00 level=INFO source=runner.go:999 msg="Server listening on 127.0.0.1:54441"
time=2025-12-03T07:36:49.496+09:00 level=INFO source=runner.go:893 msg=load request="{Operation:commit LoraPath:[] Parallel:1 BatchSize:512 FlashAttention:false KvSize:4096 KvCacheType: NumThreads:8 GPULayers:[] MultiUserCache:false ProjectorPath: MainGPU:0 UseMmap:false}"
time=2025-12-03T07:36:49.497+09:00 level=INFO source=server.go:1294 msg="waiting for llama runner to start responding"
time=2025-12-03T07:36:49.497+09:00 level=INFO source=server.go:1328 msg="waiting for server to become available" status="llm server loading model"


llama_model_loader: loaded meta data with 25 key-value pairs and 291 tensors from $HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Mistral-7B-Instruct-v0.3
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 32768
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 32768
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,32768]   = ["<unk>", "<s>", "</s>", "[INST]", "[...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,32768]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,32768]   = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  20:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 4.07 GiB (4.83 BPW)
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 2 ('</s>')
load: special tokens cache size = 771
load: token to piece cache size = 0.1731 MB
print_info: arch             = llama
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 4096
print_info: n_layer          = 32
print_info: n_head           = 32
print_info: n_head_kv        = 8
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 4
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 14336
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: model type       = 7B
print_info: model params     = 7.25 B
print_info: general.name     = Mistral-7B-Instruct-v0.3
print_info: vocab type       = SPM
print_info: n_vocab          = 32768
print_info: n_merges         = 0
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: LF token         = 781 '<0x0A>'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 48
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors:          CPU model buffer size =  4169.52 MiB
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = disabled
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.14 MiB
llama_kv_cache:        CPU KV buffer size =   512.00 MiB
llama_kv_cache: size =  512.00 MiB (  4096 cells,  32 layers,  1/1 seqs), K (f16):  256.00 MiB, V (f16):  256.00 MiB
llama_context:        CPU compute buffer size =   300.01 MiB
llama_context: graph nodes  = 1158
llama_context: graph splits = 1
time=2025-12-03T07:36:53.011+09:00 level=INFO source=server.go:1332 msg="llama runner started in 3.59 seconds"
time=2025-12-03T07:36:53.011+09:00 level=INFO source=sched.go:517 msg="loaded runners" count=1
time=2025-12-03T07:36:53.012+09:00 level=INFO source=server.go:1294 msg="waiting for llama runner to start responding"
time=2025-12-03T07:36:53.012+09:00 level=INFO source=server.go:1332 msg="llama runner started in 3.59 seconds"
=#

end # module test_testllm_tokugh_ollama_api
