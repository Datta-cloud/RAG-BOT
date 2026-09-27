# suppose we have huge amount of text data and we want to summarize it using a language model. 
# then text we have to give to ai model ... 
# agar hum sara data ek sath bhejenge toh error aayega ki data bahut bada hai.
# so for this , we divide the data into chunks and then send it to ai model for summarization.
# first un chunks ko hum kahi store karenge ..store krenge in vector database and then we will send the chunks to ai model for summarization.
# store is basis pe karenge-> first chunks ki embeddings nikalenge and then store the embeddings in vector database.

# any kind of book/pdf -> divide into chunks -> store in vector database -> then send the chunks to ai model for summarization.
# then after user ask 1 question related to that doc ->convert ques also into embedding and 
# then search in vector database for similar chunks and then send those chunks to ai model for summarization.
# 

# phase 2=query(embeddings)->retrivers->(searching)vector store->(similar chunks+query)-> prompt->llm

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Embeddings
embedding_model = OpenAIEmbeddings()

# Vector Store
vector_store = Chroma(
    persist_directory="chroma_db",  # retrive data from chroma_db
    embedding_function=embedding_model
)

# Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5 # gives near 0 means diverse results 
    }
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])

# User Query
question = "What is gradient descent?"

# Retrieve documents
docs = retriever.invoke(question)

# Create context
context = "\n\n".join(
    doc.page_content for doc in docs
)

# Create prompt
final_prompt = prompt.invoke({
    "context": context,
    "question": question
})

# Generate answer
response = llm.invoke(final_prompt)

print("\n AI Answer:")
print(response.content)