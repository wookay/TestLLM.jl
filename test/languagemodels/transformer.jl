using Jive
@useinside module test_languagemodels_transformer

using Test
using Pkg.Artifacts: artifact_path
using LanguageModels
using .LanguageModels.UnicodePlots: barplot
using .LanguageModels: RunState, transformer!, load_model, load_sentencepiece_tokenizer, softmax
using .LanguageModels: stories15M_model # ::SHA1

@test stories15M_model isa Base.SHA1

# LanguageModels.jl/src/transformer.jl  function main

format = "tinyllamas"
default_model = artifact_path(stories15M_model)
checkpoint_filename = joinpath(default_model, "stories15M.bin")
tokenizer_filename = joinpath(default_model, "tokenizer.model")
mmap = true
materialize = mmap ? identity : copy
T = Array{Float32}
config, weights = load_model(checkpoint_filename; materialize = materialize)

state = RunState(T, config)
@test state.logits isa Vector{Float32}
@test length(state.logits) == 32000
d1 = softmax(state.logits)
idxs1 = findall(d1 .> 0.01)
@test idxs1 == []

token = 2
pos = 1
transformer!(token, pos, config, state, weights) ###

d2 = softmax(state.logits)
idxs2 = findall(d2 .> 0.01)
@test idxs2 == [366, 3119, 4336, 9039]

plot_probabilities = true
tokenizer = load_sentencepiece_tokenizer(tokenizer_filename)
io = stdout
print(io, barplot(
    String[tokenizer.alphabet[i] for i in idxs2],
    d2[idxs2]
))

#=
# LanguageModels.jl/src/modeldata.jl

struct Config
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn (Feed-Forward Network) layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level)
    seq_len::Int    # max sequence length
    shared_weights::Bool
end

"current wave of activations"
@kwdef struct RunState{T, Vec<:AbstractVector{T}, Arr3<:AbstractArray{T,3}}
    x::Vec      # activation at current time stamp (dim,)
    xb::Vec     # same, but inside a residual branch (dim,)
    xb2::Vec    # an additional buffer just for convenience (dim,)
    hb::Vec     # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vec    # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vec      # query (dim,)
    k::Vec      # key (dim,)
    v::Vec      # value (dim,)
    att::Vec    # buffer for scores/attention values (seq_len,)
    logits::Vec # output logits
    # kv cache
    key_cache::Arr3   # (dim, seq_len, layer)
    value_cache::Arr3 # (dim, seq_len, layer)
end
=#

end # module test_languagemodels_transformer
