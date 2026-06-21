# AI Research Paper Assistant

An AI-powered Research Paper Assistant that uses Retrieval-Augmented Generation (RAG), FAISS, and Google Gemini to help users interact with research papers through natural language questions.

## Overview

Reading and understanding research papers can be time-consuming, especially when searching for specific information. This project simplifies the process by allowing users to upload research papers in PDF format and ask questions directly about the document. The system retrieves the most relevant content from the paper and generates accurate answers using a Large Language Model.

## Features

* Upload research papers in PDF format
* Extract text from PDF documents
* Automatic text chunking and preprocessing
* Generate semantic embeddings using Sentence Transformers
* Store embeddings in a FAISS vector database
* Retrieve relevant document sections based on user queries
* Question answering using Google Gemini
* Interactive web interface built with Streamlit

## Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI & Machine Learning

* Google Gemini API
* Sentence Transformers
* Retrieval-Augmented Generation (RAG)

### Database

* FAISS Vector Store

### Libraries

* PyPDF
* NumPy
* LangChain Text Splitters

## Project Workflow

1. User uploads a research paper in PDF format.
2. The PDF text is extracted and cleaned.
3. The extracted text is divided into smaller chunks.
4. Sentence Transformer converts text chunks into vector embeddings.
5. Embeddings are stored in a FAISS vector database.
6. User enters a question related to the research paper.
7. Relevant chunks are retrieved using semantic similarity search.
8. Retrieved context is sent to Gemini.
9. Gemini generates a context-aware answer.
10. The response is displayed through the Streamlit interface.

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/AI-Research-Paper-Assistant.git
cd AI-Research-Paper-Assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Or directly configure the API key inside the application if required.

## Running the Application

```bash
streamlit run app.py
```

After running the command, open the local URL displayed in the terminal.

## Example Questions

* What is the main objective of this research paper?
* Summarize the methodology used in the paper.
* What dataset was used in the study?
* What are the key findings of the research?
* What limitations are mentioned by the authors?
* What future work is suggested?

## Future Enhancements

* Multi-document support
* Research paper summarization
* Citation extraction
* Research paper comparison
* Chat history support
* PDF highlighting for retrieved sections
* Support for multiple LLM providers

## Project Structure

```text
AI-Research-Paper-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── data/
    └── research_papers.pdf
```

## Applications

* Academic Research
* Literature Review
* Student Learning
* Research Paper Analysis
* Knowledge Retrieval
* Scientific Document Understanding
