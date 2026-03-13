from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@app.route('/')
def home():
    return "SafeHer Backend is Running with Gemini! 🚀"


@app.route('/api/gemini', methods=['POST'])
def call_gemini():
    data = request.json
    try:
        # 1. Translate message history into Gemini's format
        gemini_history = []
        for msg in data.get('messages', []):
            # Gemini uses 'model' instead of 'assistant'
            role = "model" if msg['role'] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg['content']]})

        # 2. Initialize the Gemini model with the system instruction
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # Fast and highly capable
            system_instruction=data.get('system', '')
        )

        # 3. Set generation configuration (like max tokens)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=data.get('max_tokens', 500)
        )

        # 4. Generate the response
        response = model.generate_content(
            gemini_history,
            generation_config=generation_config
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)