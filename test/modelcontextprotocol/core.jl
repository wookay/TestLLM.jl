module test_modelcontextprotocol_core

using Test
using ModelContextProtocol: ModelContextProtocol as MCP

# from ModelContextProtocol/src/types.jl
state = MCP.ServerState()
@test state.initialized === false
@test state.running === false
@test state.last_request_id == 0
@test isempty(state.pending_requests)

#=
mutable struct ServerState
    initialized::Bool
    running::Bool
    last_request_id::Int
    pending_requests::Dict{RequestId, String}  # method name for each pending request

    ServerState() = new(false, false, 0, Dict())
end
=#

end # module test_modelcontextprotocol_core
