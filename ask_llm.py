""" This script is used to ask questions to the LLM model using the SuperDuperDB as RAG. """

import streamlit as st
from rag_superduperdb import PDFDocumentHandler
from superduperdb.ext.openai import OpenAIChatCompletion
from superduperdb import Document
import os

# Load env variables
from dotenv import load_dotenv
load_dotenv()
# or set your env variables directly
os.environ['OPENAI_API_KEY'] = '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'

# Initialize the PDF handler
handler = PDFDocumentHandler()

def get_answer_from_pdf(pdf_path, question):
    """ Get the answer to a question from a PDF document.
    Args:
        pdf_path (str): The path to the PDF file.
        question (str): The question to answer.
    Returns:
        str: The answer to the question.
    """
    # Get relevant documents from the PDF
    relevant_docs = handler.get_relevant_docs(pdf_path, question)
    db_instance = handler.get_db()
    db_collection = handler.get_collection()
    db_model = handler.get_model()
    prompt = (
        'Use the following context and answer this question about context\n'
        'Do not use any other information you might have learned about other python packages\n'
        'Only base your answer on the code snippets retrieved and provide a very concise answer\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    )
    # Create an OpenAIChatCompletion instance
    chat = OpenAIChatCompletion(identifier='gpt-3.5-turbo', prompt=prompt)
    db_instance.add(chat)
    # Define search parameters
    num_results = 5
    # Generate a response
    output, sources = db_instance.predict(
        model_name='gpt-3.5-turbo',
        input=question,
        context_select=(
            db_collection
                .like(Document({'text': question}), vector_index=f'pymongo-docs-{db_model.identifier}', n=num_results)
                .find()
        ),
        context_key='text',
    )
    
    # Prepare reference 
    links = '\n'.join([f'[{i+1}] Page {r["page"]}' for i, r in enumerate(sources)])
    refs = '\n'.join([f'{r["text"]}' for r in sources])
    ref_url = '\n'.join([f'{r["url"]}' for r in sources])
    
    return output.content, links, refs, ref_url

# Streamlit UI
st.title('PDF Document Question Answering')

# User input for PDF URLs
pdf_urls_input = st.text_area("Enter PDF URLs (one per line):")
pdf_urls = pdf_urls_input.split("\n") 
print(pdf_urls)
question = st.text_input("Enter your question:")

if pdf_urls and question:
    answer, links, refs, url = get_answer_from_pdf(pdf_urls, question)
    st.markdown("### Answer")
    st.write(answer)
    st.markdown("### Page numbers of relevant text snippets")
    st.markdown(links)
    st.markdown("### Reference Texts")
    st.write(refs)
    st.markdown("### URL")
    st.write(url)
