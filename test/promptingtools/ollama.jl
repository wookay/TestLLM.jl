module test_promptingtools_ollama

using Test
using PromptingTools: PromptingTools as PT
using .PT: OllamaSchema

schema = OllamaSchema()
@info schema

end # module test_promptingtools_ollama
