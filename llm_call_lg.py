from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

conversation_history = []

system_prompt = "You are a friendly Python tutor who explains things simply to beginners."

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
    
    conversation_history.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + conversation_history
    )
    
    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    print(f"AI: {assistant_reply}\n")