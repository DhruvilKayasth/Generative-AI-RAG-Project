from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# here we want to generate embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
  
    embedding_function = embeddings,
    persist_directory= "chroma-db"
)

retriver = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assitant. Use only the provided context to answer the question. If the answer is not present in the context, say:"I could not find the answer in the document."
         """),
        ("human", """context : {context}
         Question : 
         {question}
         """)
    ]
)

print("Rag system created")

print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break
    
    docs = retriver.invoke(query)  
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })  
    
    response = llm.invoke(final_prompt)
    
    print(f"\n AI : {response.content}")
      
      
# # data = TextLoader("notes.txt", encoding="utf-8")
# # data = PyPDFLoader("GRU.pdf")
# data = PyPDFLoader("deep-learning.pdf")
# docs = data.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunks = splitter.split_documents(docs)

# # template = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", "you are a AI that summarizes the pdf in 15 paragraph ,each page have each paragraph"),
# #         ("human", "{data}")
# #     ]
# # )

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

# # prompt = template.format_messages(data=docs[0].page_content)    #only first page
# prompt = template.format_messages(data=docs)  #full pdf sending

# result = model.invoke(prompt)

# print(result.content)

# # When you send whole pdf(500+) pages then it give error depends on model,
# # because it can't  process too much big pdfs or document ,
# # so that we use chunking , we divides our pdf into no. of chunks,
# # and then we store that chunks into vector DB ,bcz there is no point of giving that chunks into LLM
# # so first we convert our pdf text into embeddings and then store it into vetor db

# # steps
# '''
# 1. load data like pdfs, document etc.. using loaddotenv and langchain documnet loaders
# 2. take one ai model api key like gemini, mistral
# and talk with the ai model
# 3. after that we split the big data into small chunks
# 4. and then that chunks are converted into embeddings
# 5. embeddings are stored in vector DB
# 6. after that user gave some user query (que) so we also convert that query into query embeddings
# 7. then we use similarity search for matching user embeddings with vector db embeddings using indexing techniques
# ''' 