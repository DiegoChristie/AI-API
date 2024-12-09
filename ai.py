from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from tempfile import mkdtemp
import shutil
import os
import requests

model_version = "llama3.3:70b-instruct-q5_K_M"

def process_w_ai(checklist_input, user_text_input, files_input):
    pdf_folder_path = "pdf_documents/"
    #image_folder_path = "image_documents/"
    
    download_pdfs_from_urls(files_input, pdf_folder_path)
    
    all_docs = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

    print("CANTIDAD DE DOCUMENTOS: " + str(len(all_docs)))
    if (len(all_docs) > 0):
        llm = OllamaLLM(model=model_version)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

        splits = text_splitter.split_documents(all_docs)

        chroma_temp_dir = mkdtemp()

        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs = {'device':'cuda'} ), persist_directory=chroma_temp_dir)
        
        retriever = vectorstore.as_retriever()

        system_prompt = (
            "You are an assistant for checking compliance to certain regulations. "
            "The following pieces of context are the user's submission.  "
            "Your job is to determine if the user submission meets the specific standards set by the regulation."
            "\n\n"
            "User written input: " + user_text_input +
            "User input through PDF files: " + "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        results = rag_chain.invoke({"input": "Give a justification for your answer. The regulation for comparing with the user submission (context) is the following: " + checklist_input})

        shutil.rmtree(chroma_temp_dir, ignore_errors=True)
        cleanup_documents_folder(pdf_folder_path)

        return item_approval(results['answer'])
    
    else:
        llm = OllamaLLM(model=model_version)

        response = llm.invoke("Please review the following details and determine compliance: Regulation Overview: " + checklist_input +
                            " User Submission: " + user_text_input + 
                            "Determine if the user submission meets the following specific standards set by the regulation. Include a brief explanation for your decision based on the information provided.")
        
        return item_approval(response)

def item_approval(response):
    llm = OllamaLLM(model=model_version)

    response2 = llm.invoke("Is the sentiment of the evaluation positive in the following statement? If positive answer ONLY 'Positive', if not answer ONLY 'Negative'. Statement: " + response)

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

def cleanup_documents_folder(directory):
    #Delete all files in the specified directory.
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            
def download_pdfs_from_urls(url_list, folder_path):
    """
    Downloads PDF files from a list of URLs and saves them to the given folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_path = os.path.join(folder_path, f"document_{i+1}.pdf")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download {url} (status code: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")