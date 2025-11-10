module test_llama2_karpathy

using Llama2
load_karpathy_model
sample

using Test
using Pkg.Artifacts: artifact_path
using LanguageModels: stories42M_model # ::SHA1
const default_model = artifact_path(stories42M_model)

format = "tinyllamas"
checkpoint_filename = joinpath(default_model, "stories42M.bin")
tokenizer_filename = joinpath(default_model, "tokenizer.model")

# Convert the tokenizer.model into .bin file:
# https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.py
# python3 tokenizer.py --tokenizer-model=./tokenizer.model

#=
model_dir = normpath(pathof(Llama2), "..", "..")
tokenizer_bin = joinpath(model_dir, "tokenizer.bin")

model = load_karpathy_model(checkpoint_filename, tokenizer_bin)
sample(model, "Julia is"; stop_on_special_token=false, bos_token=false)
=#

end # module test_llama2_karpathy
