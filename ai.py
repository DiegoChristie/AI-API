from langchain_ollama import OllamaLLM


def process_w_string(user_input, checklist_input):
    llm = OllamaLLM(model="llama3.1")

    response = llm.invoke("Please review the following details and determine compliance: Regulation Overview: " + checklist_input +
                          " User Submission: " + user_input + 
                          "Determine if the user submission meets the following specific standards set by the regulation. Include a brief explanation for your decision based on the information provided.")
    
    return item_approval(response)

def item_approval(response):
    llm = OllamaLLM(model="llama3.1")

    response2 = llm.invoke("Is the sentiment positive in the following statement? If positive answer ONLY 'Positive', if not answer ONLY 'Negative'. Statement: " + response)

    decision = parse_response(response2)

    return [decision,response]

def parse_response(response):
    # Convert response to lower case for case-insensitive comparison
    lower_response = response.lower()
    decision = ''
    # Check for 'positive' or 'negative' in the response
    if 'positive' in lower_response:
        decision = 'Positive'
    elif 'negative' in lower_response:
        decision = 'Negative'
    else:
        decision = 'Unclear'

    return decision

def process_w_pdf(checklist_input):
    pass