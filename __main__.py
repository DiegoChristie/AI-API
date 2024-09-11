from flask import Flask, request, jsonify
from ai import *
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'documents'

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
        # Check if the request contains a file
        if 'pdf_file' not in request.files:
            return "No file part", 400

        # Get the file object
        file = request.files['pdf_file']
        
        # Get the file name
        file_name = file.filename

        timestamp = str(int(time.time()))
        file_name_with_timestamp = f"{timestamp}_{file_name}"
        
        # Read the binary data from the file
        #pdf_binary = file.read()

        checklist_input = request.form.get('checklist_input')
        # Define the path to save the file (inside the uploads folder)

        save_path = os.path.join(UPLOAD_FOLDER, file_name_with_timestamp)
        
        # Save the file to the specified folder
        file.save(save_path)

        
        # Process the prompt (this is where you'd integrate your AI processing logic)
        response = process_w_pdf(checklist_input, file_name_with_timestamp)

        # Return the response in JSON format
        return jsonify({'sentiment': response[0],'response': response[1]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)