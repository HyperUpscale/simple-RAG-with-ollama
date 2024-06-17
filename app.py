import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, GitLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import Client

# Define the client for the Ollama service on the local machine
client = Client(host='http://localhost:11434')

# Load the local chat model
model_local = ChatOllama(model="phi3")

# Function to load the data source, process it into documents, and create a vector store
def vector_input(source_type, source_url):
    if source_type == "CSV":
        # Load the CSV file into a DataFrame
        data = pd.read_csv(source_url)
        print(f"Loaded CSV file: {source_url}")
        print(data.head(10))

        # List to store the document objects
        documents = []
        # Iterate over the rows in the DataFrame
        for index, row in data.iterrows():
            # Concatenate all column values into a single text block
            content = " ".join(map(str, row.values))
            # Create a Document object with the concatenated content and metadata
            document = Document(page_content=content, metadata={"row_index": index})
            # Add the Document to the list
            documents.append(document)

    elif source_type == "URL":
        loader = WebBaseLoader(source_url)
        documents = loader.load()

    elif source_type == "GitHub":
        loader = GitLoader(source_url)
        documents = loader.load()

    # Ensure that the list contains only Document objects
    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError("Expected list of Document objects.")

    # Use RecursiveCharacterTextSplitter to split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter()
    doc_splits = text_splitter.split_documents(documents)

    # Create embeddings for the split documents and store them in a Chroma vector store
    print("Starting nomic-embed-text embedding")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    return retriever

# Function to process user input and run the RAG process
def process_input(source_type, source_url, question):
    retriever = vector_input(source_type, source_url)

    print("\n########\nAfter RAG\n")

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)

# Define a Gradio interface with source selection and input field
with gr.Blocks() as demo:
    gr.Markdown("## Document Query with Ollama")
    source_type = gr.Dropdown(
        choices=["CSV", "URL"],
        label="Source Type",
        value="CSV",
    )
    source_url = gr.Textbox(
        label="Source URL/File Path",
        placeholder="Enter the URL or file path",
    )
    question = gr.Textbox(label="Question", placeholder="Enter your question")
    output = gr.Textbox(label="Answer")

    submit_button = gr.Button("Submit")
    submit_button.click(
        process_input,
        inputs=[source_type, source_url, question],
        outputs=output,
    )

iface = demo.launch(share=True)
