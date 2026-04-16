module test_modelcontextprotocol_integration

using Test
using ModelContextProtocol: ModelContextProtocol as MCP
using .MCP: Server, ServerConfig, ResourceCapability, ToolCapability, PromptCapability
using .MCP: MCPPrompt, PromptArgument, PromptMessage, TextContent
using .MCP: RequestContext, ListPromptsParams

# from ModelContextProtocol/test/integration/full_server.jl

config = ServerConfig(
    name = "test-server",
    version = "1.0.0",
    capabilities = [
        ResourceCapability(list_changed = true, subscribe = true),
        ToolCapability(list_changed = true),
        PromptCapability(list_changed = true)
    ]
)
server = Server(config)

test_prompt = MCPPrompt(
    name = "test-prompt",
    description = "A test prompt",
    arguments = [PromptArgument(name = "arg1", description = "Test arg", required = true)],
    messages = [PromptMessage(
        content = TextContent(type = "text", text = "Test prompt with {arg1}"),
        role = MCP.user
    )]
)

MCP.register!(server, test_prompt)

ctx = RequestContext(server = server, request_id = 1)
handle_result = MCP.handle_list_prompts(ctx, ListPromptsParams())
@test handle_result.response.result["prompts"][1]["name"] == test_prompt.name

end # module test_modelcontextprotocol_integration
