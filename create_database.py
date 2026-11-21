# load the pdf
# split into chunks
# create the embeddings
# store into chromadb


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# load the pdf
load_dotenv()

data = PyPDFLoader("deep-learning.pdf")
docs = data.load()

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

# create the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vectorstore
vectorstore = Chroma.from_documents(
    documents = chunks, #here we don't send docs fully bcz it is too big,
                        # so we send chunks to the vectorstore which are made by splitting
    embedding= embeddings,
    persist_directory= "chroma-db"
)
