from flask import Flask, request, jsonify
from ai import *

app = Flask(__name__)

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    if request.method == 'POST':
        # Extract prompt from the received JSON
        data = request.json
        user_input = data.get('user_input', '')
        checklist_input = data.get('checklist_input', '')

        # Process the prompt (this is where you'd integrate your AI processing logic)
        response = process_ai(user_input, checklist_input)

        # Return the response in JSON format
        return jsonify({'sentiment': response[0],'response': response[1]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)