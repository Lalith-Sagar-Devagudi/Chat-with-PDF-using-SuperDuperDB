# Chatbot Using SuperDuperDB with RAG and LLM

This chatbot application allows you to ask questions about your documentation using SuperDuperDB as a Retrievable Answer Generator (RAG) and a Large Language Model (LLM). The application leverages MongoDB for database management and OpenAI's API for generating responses. Follow the instructions below to set up and run the chat application.

## Prerequisites

Before you begin, ensure you have the following installed:
- MongoDB Community Edition
- Python 3.8 or higher

## Setup Instructions

### Step 1: Start MongoDB

1. Start the MongoDB Community Edition using the following command:
   ```shell
   brew services start mongodb-community
   ```
2. Connect to your MongoDB instance using the Mongo Shell with:
   ```shell
   mongosh mongodb://localhost:27017
   ```

### Step 2: Install Dependencies

Install all the necessary dependencies by running the following command in your project's root directory:
```shell
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

Set the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `MONGODB_URI`: Your MongoDB connection URI.

Example:
```shell
export OPENAI_API_KEY='your_openai_api_key_here'
export MONGODB_URI='mongodb://localhost:27017'
```

#### About the file `ask_llm.py`
This script integrates a PDF document question-answering feature with a Streamlit interface, leveraging OpenAI's language models for generating answers based on the content extracted from PDF documents:

- **Environment Setup**: Utilizes `dotenv` for environment variable management, directly setting the OpenAI API key within the script for simplicity, though in production, it's recommended to keep sensitive keys in environment variables for security.

- **PDF Document Handler**: Initializes a `PDFDocumentHandler` object for processing PDF documents, extracting relevant text snippets as potential context for answering questions.

- **Question-Answering Function**: `get_answer_from_pdf` function takes a PDF path and a question, retrieves relevant document snippets, and uses these snippets as context for generating answers through an OpenAI model. It showcases an advanced use of MongoDB for storing and querying document snippets, using a custom model and collection setup.

- **Streamlit Interface**: Provides a user-friendly web interface for inputting PDF URLs and questions. The script processes these inputs, queries the database for relevant context, and displays the model-generated answer along with references to the context (e.g., page numbers, text snippets, URLs) that informed the answer.

- **Integration of Components**: Demonstrates a sophisticated integration of PDF processing, database operations, AI model querying, and web interface management, illustrating a full-stack Python application capable of complex data processing and user interaction.

#### About the file `rag_superduperdb.py`
The `PDFDocumentHandler` class in the script provides an end-to-end solution for processing PDF documents, extracting text, chunking it into manageable pieces, embedding these pieces using a machine learning model, and then storing the chunks in a MongoDB database for later retrieval. Here's a detailed breakdown:

- **Initialization**: It sets up a connection to MongoDB and a path for storing artifacts, initializing the database with SuperDuperDB and a semantic chunker for text processing.

- **Database and Model Setup**: The class initializes a SentenceTransformer model for text embeddings, sets a unique name for document collections, and adds a vector index to enhance search capabilities within the MongoDB collection.

- **PDF Processing**: Methods are provided to download PDFs from URLs, extract text with page numbers, chunk the text semantically, and insert these chunks into the database as documents.

- **Querying**: It includes a method to query the database for documents relevant to a given search term, utilizing vector embeddings for semantic search, returning the most relevant chunks based on the search term.

- **Integration and Utility**: The class offers utility methods to retrieve the database instance, collection, and model for external use, facilitating integration with other components of the application.

### Step 4: Run the Streamlit Interface

Launch the chatbot interface using Streamlit by executing:
```shell
streamlit run ask_llm.py
```

## User Interface

The user interface for the chat application is accessible through Streamlit. Below is a preview of what to expect:

https://github.com/Lalith-Sagar-Devagudi/Chat-with-PDF-using-SuperDuperDB/assets/40135491/c43f39fc-6508-47fa-bd73-deae5e32bac7
