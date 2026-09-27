# load pdf
# split into chunks
# create the embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load PDF
data = PyPDFLoader(
    "document loaders/deep-learning-material-dept-ece-ase-blr-1.pdf"
)

docs = data.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# 3. Create embeddings
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# 4. Store embeddings in Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)