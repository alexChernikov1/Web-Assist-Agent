#load_dotenv()
#OLLAMA_HOST:http://127.0.0.1:11434
url="http://127.0.0.1:11434/api/chat"

payload={
    "model":"deepseek-r1:1.5b",
    "messages":[{"role":"user","content":"What is Python"}]
}

api_key = os.getenv("API_KEY")
print(f"My API key is: {api_key}")  # ✅ This will print the string from .env
# Please install OpenAI SDK first: `pip3 install openai`



client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)