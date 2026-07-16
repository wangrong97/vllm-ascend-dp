# curl http://localhost:9001/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "dsv",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Who are you?"
#             }
#         ],
#         "max_tokens": 256,
#         "temperature": 0
#     }'

curl http://localhost:9002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv",
        "messages": [
            {
                "role": "user",
                "content": "\nEvery morning, Aya does a $9$ kilometer walk, and then finishes at the coffee shop. One day, she walks at $s$ kilometers per hour, and the walk takes $4$ hours, including $t$ minutes at the coffee shop. Another morning, she walks at $s+2$ kilometers per hour, and the walk takes $2$ hours and $24$ minutes, including $t$ minutes at the coffee shop. This morning, if she walks at $s+\\frac12$ kilometers per hour, how many minutes will the walk take, including the $t$ minutes at the coffee shop?\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            }
        ],
        "max_tokens": 30000,
        "temperature": 0
    }'