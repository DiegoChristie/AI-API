from flask import Flask, request, jsonify
from ai import *
import base64

app = Flask(__name__)

@app.route('/string_compliance_check', methods=['POST'])
def string_compliance_check():
    if request.method == 'POST':
        # Extract prompt from the received JSON
        data = request.json
        user_input = data.get('user_input', '')
        checklist_input = data.get('checklist_input', '')

        # Process the prompt (this is where you'd integrate your AI processing logic)
        response = process_w_string(user_input, checklist_input)

        # Return the response in JSON format
        return jsonify({'sentiment': response[0],'response': response[1]})


@app.route('/pdf_compliance_check', methods=['POST'])
def pdf_compliance_check():
    if request.method == 'POST':
        # Extract prompt from the received JSON
        data = request.json
        checklist_input = data.get('checklist_input', '')

        pdf_data = data.get('file')  # Extract Base64 encoded PDF data
        pdf_filename = data.get('filename', 'uploaded.pdf')  # Extract filename, default to 'uploaded.pdf'

        if pdf_data:
            # Decode the Base64 string to binary data
            pdf_binary = base64.b64decode(pdf_data)

            # Save the PDF to a file (optional)
            with open(pdf_filename, 'wb') as pdf_file:
                pdf_file.write(pdf_binary)

        # Process the prompt (this is where you'd integrate your AI processing logic)
        response = process_w_pdf(checklist_input, pdf_filename)

        # Return the response in JSON format
        return jsonify({'sentiment': response[0],'response': response[1]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)