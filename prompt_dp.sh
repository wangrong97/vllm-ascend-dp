curl http://localhost:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'