from langchain_community.document_loaders import WebBaseLoader

url = "https://cricket07.dhaiwat.com/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)

#Web Page loader