module test_zana_agent

using Test
using Zana

config = ZanaConfig(
    ollama = OllamaConfig(model="mistral")
)

agent = ZanaAgent(config, ".")

@test agent.config.ollama.host == "http://localhost:11434"

end # module test_zana_agent
