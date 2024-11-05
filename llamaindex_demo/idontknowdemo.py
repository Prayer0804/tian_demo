from openai import OpenAI

client = OpenAI(
    base_url="https://api.gptsapi.net/v1",
    api_key="xxx"
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "you are a helpful assistant."},
        {
            "role": "user",
            "content": "i need some emoji"
        }

    ]
)

print(response.choices[0].message.content)
