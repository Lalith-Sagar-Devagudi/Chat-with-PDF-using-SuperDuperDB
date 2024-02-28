""" This module contains the PDFDocumentHandler class, which is used to process PDFs and query the processed documents. """

import os
from superduperdb import superduper, Document, Model, Listener, VectorIndex, vector
from superduperdb.backends.mongodb import Collection
from langchain_experimental.text_splitter import SemanticChunker
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import random
import string
import requests

class PDFDocumentHandler:
    """A class for processing PDFs and querying the processed documents."""
    def __init__(self, mongodb_uri="mongomock://test", artifact_store_path='./data/'):
        """
        Initializes the PDFDocumentHandler object.

        Args:
            mongodb_uri (str): The URI for the MongoDB database.
            artifact_store_path (str): The path to the artifact store directory.

        """
        self.mongodb_uri = mongodb_uri
        self.artifact_store_path = artifact_store_path
        self.db = self.initialize_database()
        self.text_splitter = SemanticChunker(HuggingFaceEmbeddings())


    def initialize_database(self):
        """ Initializes the SuperDuperDB database."""
        db = superduper(self.mongodb_uri, artifact_store=f'filesystem://{self.artifact_store_path}')
        return db

    def initialize_model(self):
        """ Initializes the SentenceTransformer model.
        Returns:
            Model: The initialized model.
        """
        model = Model(
            identifier='all-MiniLM-L6-v2',
            object=SentenceTransformer('all-MiniLM-L6-v2'),
            encoder=vector(shape=(384,)),
            predict_method='encode',
            postprocess=lambda x: x.tolist(),
            batch_predict=True,
        )
        return model

    def set_collection_name(self):
        """Sets the collection name .
        Args:
            pdf_path (str): The path to the PDF file.
        """
        # random string as collection name
        collection_name = ''.join(random.choices(string.ascii_lowercase, k=7))
        self.doc_collection = Collection(collection_name)

    def add_vector_index(self):
        """Adds a vector index to the database."""
        self.db.add(
            VectorIndex(
                identifier=f'pymongo-docs-{self.model.identifier}',
                indexing_listener=Listener(
                    select=self.doc_collection.find(),
                    key='text',
                    model=self.model,
                    predict_kwargs={'max_chunk_size': 1000},
                ),
            )
        )

    def download_pdf(self, pdf_url):
        """ Downloads a PDF from a URL and saves it to the artifact store.
        Args:
            pdf_url (str): The URL of the PDF.
        """
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        return response.content

    def process_pdf(self, pdf_urls: list):
        """Processes the PDF file and inserts the documents into the database.
        Args:
            pdf_path (str): The path to the PDF file.
        """
        self.set_collection_name()
        self.model = self.initialize_model()
        self.add_vector_index()
        for pdf_url in pdf_urls:
            all_text_with_markers = self.extract_text_with_page_numbers_from_url(pdf_url)
            chunks_with_pages = self.chunk_text(all_text_with_markers, pdf_url)
        self.insert_documents(chunks_with_pages)


    def extract_text_with_page_numbers_from_url(self, pdf_url):
        """
        Extracts text along with page numbers from a PDF URL.
        """
        all_text_with_markers = []
        pdf_bytes = self.download_pdf(pdf_url)  # Download PDF content as bytes
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:  # Open the PDF directly from bytes
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                all_text_with_markers.append((text, page_num))
        return all_text_with_markers

    def chunk_text(self, all_text_with_markers, url):
        """Chunks the text and assigns the page numbers to the chunks.
        Args:
            all_text_with_markers (list): A list of tuples containing the text and the page number. 
        Returns:
            list: A list of dictionaries containing the text and the page number.
        """
        chunks_with_pages = []
        for text, page_num in all_text_with_markers:
            chunks = self.text_splitter.create_documents([text])
            for chunk in chunks:
                chunks_with_pages.append((chunk, page_num))
        return [{"text": chunk[0].page_content, "page": chunk[1], "url": url} for chunk in chunks_with_pages]

    def insert_documents(self, chunks_with_pages):
        """Inserts the documents into the database.
        Args:
            chunks_with_pages (list): A list of dictionaries containing the text and the page number.
        """
        self.db.execute(self.doc_collection.insert_many([Document(r) for r in chunks_with_pages]))

    def query_documents(self, search_term, num_results=1):
        """Queries the documents in the database.
        Args:
            search_term (str): The search term.
            num_results (int): The number of results to return.
        Returns:
            list: A list of dictionaries containing the search results.
        """
        result = self.db.execute(
            self.doc_collection
                .like(Document({'text': search_term}), vector_index=f'pymongo-docs-{self.model.identifier}', n=num_results)
                .find()
        )
        return sorted(result, key=lambda r: -r['score'])

    def get_relevant_docs(self, pdf_list, search_term):
        """Processes the PDF file and queries the documents in the database.
        Args:
            pdf_path (str): The path to the PDF file.
            search_term (str): The search term.
        Returns:
            list: A list of dictionaries containing the search results.
        """
        self.process_pdf(pdf_list)
        return self.query_documents(search_term)
    
    def get_db(self):
        """Returns the database instance for external use.
        Returns:
            superduper: The database instance.
        """
        # Returns the database instance for external use
        return self.db
    def get_collection(self):
        """Returns the document collection instance for external use.
        Returns:
            Collection: The document collection instance.
        """
        # Returns the document collection instance for external use
        return self.doc_collection
    def get_model(self):
        """Returns the model instance for external use.
        Returns:
            Model: The model instance.
        """
        # Returns the model instance for external use
        return self.model
