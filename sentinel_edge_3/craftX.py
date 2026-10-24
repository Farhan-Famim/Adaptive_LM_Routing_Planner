import requests

#**********************************
CRAFTX_API_KEY = "cc-d2ad7d21d49d4cbd967feddc0d901332"
#**********************************

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break

    response = requests.post(
        "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CRAFTX_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "DeepSeek V4 Pro",
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 512,
        },
    )

    result = response.json()
    print("Bot:", result["choices"][0]["message"]["content"])