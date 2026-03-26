module test_modelcontextprotocol_tool

using Test
using ModelContextProtocol: ModelContextProtocol as MCP

# from ModelContextProtocol/test/features/tools.jl

# Define a tool - now with simplified return
julia_version_tool = MCP.MCPTool(
    name = "julia_version",
    description = "Get the Julia version used to run this tool",
    parameters = [],
    # Return Dict directly - it will be automatically converted to TextContent
    handler = params -> Dict("version" => string(VERSION)),
    return_type = MCP.TextContent  # Explicitly expect single TextContent
)

config = MCP.ServerConfig(
    name = "test-server",
    version = "1.0.0"
)
server = MCP.Server(config)

MCP.register!(server, julia_version_tool)

ctx = MCP.RequestContext(
    server = server,
    request_id = 1
)

params = MCP.CallToolParams(
    name = "julia_version",
    arguments = nothing
)

result = MCP.handle_call_tool(ctx, params)
@test result isa MCP.HandlerResult
@test result.response.result.content[1]["text"] == string("{", repr("version"), ":", repr(string(VERSION)), "}")

end # module test_modelcontextprotocol_tool
