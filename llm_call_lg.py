from google import genai

client = genai.Client(api_key = 'AIzaSyDvlNpjvVp0mqT0XPvQrgZgHfWT3SuR674')

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='who is the ceo of apple?'
)

print(response.text)