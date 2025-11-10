module test_languagemodels_transformer

using Test
using Pkg.Artifacts: artifact_path
using LanguageModels
using .LanguageModels: RunState, transformer!, load_model
using .LanguageModels: stories15M_model # stories15M_model::SHA1

format = "tinyllamas"
default_model = artifact_path(stories15M_model)
checkpoint_filename = joinpath(default_model, "stories15M.bin")
mmap = true
materialize = mmap ? identity : copy
T = Array{Float32}
config, weights = load_model(checkpoint_filename; materialize = materialize)
state = RunState(T, config)
token = 2
pos = 1
transformer!(token, pos, config, state, weights)

end # module test_languagemodels_transformer
