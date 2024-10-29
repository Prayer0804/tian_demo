from openai import OpenAI

client = OpenAI(
    base_url="https://api.gptsapi.net/v1",
    api_key="sk-LGZ06ac207a9329843ef8cfe2ed2ea0e47b246960c0GdhZP"
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "you are a helpful assistant."},
        {
            "role": "user",
            "content": "give me a five"
        }

    ]
)

print(response.choices[0].message)
