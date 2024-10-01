from flask import Flask, request, jsonify
from ai import *
import os
import time

app = Flask(__name__)

UPLOAD_PDF_FOLDER = 'pdf_documents'
UPLOAD_IMAGE_FOLDER = 'image_documents'

@app.route('/ai_compliance_check', methods=['POST'])
def pdf_compliance_check():
    if request.method == 'POST':
        data = request.json
        user_text_input = data.get('user_text_input', '')
        checklist_input = data.get('checklist_input', '')
        files_input = data.get('files_input', '')
        
        print(files_input)

        response = process_w_ai(checklist_input, user_text_input, files_input)

        # Return the response in JSON format
        return jsonify({'sentiment': response[0],'response': response[1]})
    
@app.route('/ai_compliance_check_test_positive', methods=['POST'])
def ai_compliance_check_test_positive():
    if request.method == 'POST':
        # Return the response in JSON format
        return jsonify({'response': 'todo gucci','sentiment': 'Positive'})
    
@app.route('/ai_compliance_check_test_negative', methods=['POST'])
def ai_compliance_check_test_negative():
    if request.method == 'POST':
        # Return the response in JSON format
        return jsonify({'response': 'todo malucci','sentiment': 'Negative'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)