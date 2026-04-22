# https://www.mindstudio.ai/blog/how-to-run-gemma-4-locally-ollama
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [
      {"role": "user", "content": "Explain how transformers work in two sentences."}
    ]
  }'

# {"id":"chatcmpl-823","object":"chat.completion","created":1776410125,"model":"gemma4","system_fingerprint":"fp_ollama","choices":[{"index":0,"message":{"role":"assistant","content":"Transformers process sequential data by using a multi-headed attention mechanism, allowing them to weigh the importance of every word in relation to every other word to build deep contextual understanding. This architecture then uses those derived relationships to model complex linguistic patterns and generate coherent, context-aware outputs in sequence.","reasoning":"Plan: Explain the core mechanism (attention) and the overall function/structure of transformers within a strict two-sentence limit.\n\nSentence 1: Focus on the input processing and the key innovation (Attention).\nSentence 2: Focus on the output generation and overall role (contextual understanding/sequence modeling).\n\nReview against constraints: Two sentences. Clear explanation.\n\nSelf-Correction/Refinement: Ensure the explanation is high-level but accurate. I must mention \"attention\" as that is the revolutionary component."},"finish_reason":"stop"}],"usage":{"prompt_tokens":23,"completion_tokens":167,"total_tokens":190}}
