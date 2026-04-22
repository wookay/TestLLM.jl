# ModelContextProtocol/examples/simple_http_server.jl

curl -X POST http://localhost:3000/ -H 'Content-Type: application/json' \
   -H 'MCP-Protocol-Version: 2025-06-18' \
   -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'

# {"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"simple-streamable-http-server","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true},"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true}},"protocolVersion":"2025-06-18","instructions":""}}


curl -X POST http://localhost:3000/ -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{"jsonrpc":"2.0", "method":"tools/call", "params":{"name":"echo", "arguments":{"message":"Hello MCP"} },"id":1}'

# {"jsonrpc":"2.0","id":1,"result":{"content":[{"text":"Echo: Hello MCP","type":"text"}],"is_error":false}}
