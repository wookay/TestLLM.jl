module test_modelcontextprotocol_transports

using Test
using ModelContextProtocol: ModelContextProtocol as MCP

# from ModelContextProtocol/test/transports/test_stdio.jl
transport = MCP.StdioTransport()
@test transport.input === stdin
@test transport.output === stdout
@test transport.connected

#=
help?> MCP.StdioTransport
  StdioTransport(; input::IO=stdin, output::IO=stdout)

  Transport implementation using standard input/output streams. This is the
  default transport for local MCP server processes.

  Fields
  ≡≡≡≡≡≡

  • input::IO: Input stream for reading messages (default: stdin)
  • output::IO: Output stream for writing messages (default: stdout)
  • connected::Bool: Connection status (always true for stdio)
=#

end # module test_modelcontextprotocol_transports
