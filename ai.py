from langchain_ollama import OllamaLLM

def process_ai(user_input, checklist_input):
    llm = OllamaLLM(model="llama3.1")

    response = llm.invoke("With '"+ checklist_input + "' being the question or statement, is the following answer strictly related?: " + user_input)

    return item_approval(response)


def item_approval(response):
    llm = OllamaLLM(model="llama3.1")

    response2 = llm.invoke("Is the sentiment positive in the following statement? If positive answer ONLY 'Yes', if not answer ONLY 'No'. Statement: " + response)

    return [response2,response]