#
# ollama_api
# code from https://gist.github.com/tokugh/497afc0f363727bd0b34456807fb8a32
#
# module TestLLM

## 基本設定

using HTTP, JSON3, Base64
const DICT = Dict{Symbol,Any}

const MODEL = (mistral = "mistral:latest", phi3="phi3:latest", llama3="llama3:latest")
#const MODEL = (phi3="phi3:latest",llama3="llama3:latest")
# @info "Default model is: $(MODEL[1])"

@enum Role user assistant


#
# ollama_init1
#

## HTTP Request/Response
const base_url = get(ENV,"OLLAMA_BASE_URL","http://localhost:11434")

url(parts...) = joinpath(base_url,parts...)
headers() = []

function caller(callee::Symbol)
    for st in stacktrace()
        hasproperty(st,:func) || continue
        f = st.func
        f != callee || continue
        f = string(f)
        startswith(f,"_") || continue
        return f
    end
    return nothing
end

function _request(method,url_options=[],headers_options=[],body=[];kwargs...)
    # @info caller(:_request)
    get(body,:stream,true) && return _request_streaming(method,url_options,headers_options,body;kwargs...)
    resp = let url = url(url_options...), headers = headers(headers_options...)
        if isempty(body)
            HTTP.request(method,url,headers;kwargs...)
        else
            HTTP.request(method,url,headers,body=JSON3.write(body);kwargs...)
        end
    end
    json = JSON3.read(resp.body)
    return json
end

function _request_streaming(method,url_options,headers_options,body;kwargs...)
    # @info "streaming: true"
    buf = PipeBuffer()
    t = @async let url = url(url_options...), headers = headers(headers_options...)
        if isempty(body)
            HTTP.request(method,url,headers;kwargs...,response_stream=buf)
        else
            HTTP.request(method,url,headers,body=JSON3.write(body);kwargs...,response_stream=buf)
        end
    end
    errormonitor(t)

    function sse_receiver(c::Channel)
        chunk = ""
        while true
            if eof(buf)
                sleep(0.125)
            else
                chunk = readline(buf,keep=true)
                if true
                    json_part = JSON3.read(chunk)
                    put!(c,json_part)
                    json_part.done && break
                    chunk = ""
                end
            end
        end
    end

    out = stdout

    chunks = Channel(sse_receiver)
    msg_json = JSON3.Object()
    parts = String[]
    for json in chunks
        if !json.done
            text = hasproperty(json,:response) ? json.response : json.message.content
            write(out,text)
            push!(parts,text)
        else
            write(out,"\n")
            msg_dict = copy(json)
            if hasproperty(json,:response)
                msg_dict[:response] = join(parts)
            else
                msg_dict[:message][:content] = join(parts)
            end
            msg_json = msg_dict |> JSON3.write |> JSON3.read
        end
    end

    wait(t)
    return msg_json
end

## APIの関数化
function _generate_a_completion(model,dict)
    dict[:model] = model
    json = _request(:POST,["api","generate"],[],dict)
end

function _generate_a_chat_completion(model,dict)
    dict[:model] = model
    json = _request(:POST,["api","chat"],[],dict)
end


#
# ollama_init2
#

## 抽象化
macro deligate_properties(T,dst)
    quote
        function Base.getproperty(obj::$T, name::Symbol)
            if hasfield($T, name)
                return getfield(obj, name)
            else
                return getfield(obj, $dst)[name]
            end
        end
        function Base.setproperty!(obj::$T, name::Symbol, x)
            if hasfield($T, name)
                return setfield!(obj, name, x)
            else
                modify(obj, name=>x)
                return getfield(obj,$dst)[name]
            end
        end
    end
end

abstract type OllamaObject end

### Message
struct Message <: OllamaObject
    json::JSON3.Object
end
@deligate_properties Message :json

### Thread
struct Thread
    messages::Vector{Message}
end
first80(str) = length(str) ≤ 80 ? str : first(str,80)*"…"
function Base.show(io::IO, thread::Thread)
    print(io,"[\n")
    for (i,msg) in enumerate(thread.messages)
        print(io,"$i: ",msg.message.role,"\n    ")
        print(io,first80(JSON3.write(msg.message.content)))
        print(io,"\n")
    end
    print(io,"]; ")
end

### State
@kwdef struct State
    thread::Thread = Thread()
    kwargs::DICT = DICT(:model=>MODEL[1] #=, :system=>"you are a helpful assistant" =#)
end
function Base.show(io::IO, state::State)
    print(io,"State(thread=")
    print(io,state.thread)
    join(io,(":$k=>$(first80(repr(v)))" for (k,v) in getfield(state,:kwargs)),", ")
    print(io,"\n)\n")
end

modify(state::State, (;first,second)::Pair) = state.kwargs[first] = second
@deligate_properties State :kwargs

## APIとのブリッジ
function generate_a_complition(message::Message; model, kwargs...)
    dict = DICT(pairs(kwargs)...)
    dict[:prompt] = message.content
    json = _generate_a_completion(model,dict)
    return json
end

function generate_a_chat_completion(thread::Thread; model, kwargs...)
    messages = map(thread.messages) do message
        (role=message.message.role,content=message.message.content)
    end
    dict = DICT(:messages=>messages,pairs(kwargs)...)
    json = _generate_a_chat_completion(model,dict)
    return Message(json)
end

function generate_a_chat_completion(state::State; model, kwargs...)
    generate_a_chat_completion(state.thread;model,state.kwargs...,kwargs...)
end


#
# ollama_api
#

## ユーティリティ
Thread() = Thread(Message[])
Thread(state::State) = getfield(state,:thread)

list(::Type{Message},thread::Thread) = getfield(thread,:messages)
list(::Type{Message},state::State) = list(Message,Thread(state))

function chat1(text::String; model=MODEL[1], kwargs...)
    json = (content=text,) |> JSON3.write |> JSON3.read
    message = Message(json)
    message = generate_a_complition(message; model=model, kwargs...)
    get(kwargs,:stream,true) || println(message.response)
    return message
end
function chat1(state::State, text::String; kwargs...)
    thread = state.thread
    messages = list(Message,thread)

    json = (message=(role=user,content=text),) |> JSON3.write |> JSON3.read
    message = Message(json)
    push!(messages,message)

    message = chat1(text; state.kwargs..., kwargs...)
    state.context = message.context

    json = (message=(role=assistant,content=message.response),) |> JSON3.write |> JSON3.read
    message = Message(json)
    push!(messages,message)
    return
end

function chat(thread::Thread, text::String; model=MODEL[1], kwargs...)
    messages = list(Message,thread)

    json = (message=(role=user,content=text),) |> JSON3.write |> JSON3.read
    message = Message(json)
    push!(messages,message)

    message = generate_a_chat_completion(thread; kwargs...,model=model)
    push!(messages,message)

    get(kwargs,:stream,true) || println(messages[end].message.content)
    return
end
chat(state::State, text::String; kwargs...) = chat(state.thread,text;state.kwargs...,kwargs...)
chat(text::String; kwargs...) = chat(State(),text;kwargs...)

# Example
"""
```julia-repl
julia> state = State()
julia> state.system = "あなたはとても粗野なアシスタントです。"
julia> state.model = "elyza"
julia> chat1(state,"こんにちは")
[ Info: streaming: true
うるせー、こんにちわ！
```
"""
State


if PROGRAM_FILE == "ollama_api.jl"
    state = State()
    state.model = "mistral"
    chat1(state, "こんにちは")
end

# module TestLLM
