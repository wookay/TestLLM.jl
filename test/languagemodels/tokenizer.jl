using Jive
@useinside module test_languagemodels_tokenizer

using Test
using Pkg.Artifacts: artifact_path
using LanguageModels: LanguageModels
using .LanguageModels: DigramEncodingTokenizer, load_sentencepiece_tokenizer
using .LanguageModels: stories15M_model # ::SHA1

format = "tinyllamas"
default_model = artifact_path(stories15M_model)
tokenizer_filename = joinpath(default_model, "tokenizer.model")

tokenizer = load_sentencepiece_tokenizer(tokenizer_filename)

@test tokenizer isa DigramEncodingTokenizer{String, Float32}
@test length(tokenizer.alphabet) == 32000

idxs2 = [366, 3119, 4336, 9039]
@test [" L", " One", " Tom", " Once"] == tokenizer.alphabet[idxs2]
@test [-106.0, -2859.0, -4076.0, -8779.0] == tokenizer.scores[idxs2]

#=
# LanguageModels.jl/src/tokenizer.jl

struct DigramEncodingTokenizer{T, S<:Real}
    alphabet::Vector{T}
    scores::Vector{S}
    output::Vector{T}
end

=#

end # module test_languagemodels_tokenizer
