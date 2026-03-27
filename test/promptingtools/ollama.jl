module test_promptingtools_ollama

using Test
using PromptingTools: PromptingTools as PT

schema = PT.OllamaSchema()
prompt = "hi"

on_ci = haskey(ENV, "CI")
if !on_ci
conversation = PT.aigenerate(schema,
                             prompt ;
                             verbose = true,
                             model = "mistral",
                             return_all = true)
@test conversation[1] isa PT.SystemMessage
@test conversation[2] isa PT.UserMessage
@test conversation[3] isa PT.AIMessage
end # if

end # module test_promptingtools_ollama
