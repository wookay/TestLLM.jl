using Zana
config = ZanaConfig(
    ollama = OllamaConfig(model="mistral")
)
agent = ZanaAgent(config, ".")
@info agent
