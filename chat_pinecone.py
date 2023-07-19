import os
import sys
import constants

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
import pinecone

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Pinecone configuration
pinecone.init(
    api_key=constants.PINECONE_API_KEY,
    environment=constants.PINECONE_ENVIRONMENT
)

index_name = "gpt-info"
# index = pinecone.Index("gpt-info")
# pinecone.create_index(index_name=pinecone_index_name)

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# loader = DirectoryLoader("data/")
# data = loader.load()
# # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
# docs = text_splitter.split_documents(data)

interface = Pinecone.from_existing_index(
    index_name,
    embedding=OpenAIEmbeddings()
)

# interface = Pinecone.from_documents(docs, embedding=OpenAIEmbeddings(), index_name=index_name)

qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, streaming=True),
    retriever=interface.as_retriever(search_kwargs={"k": 1})
)

chat_history = []
while True:
    if not query:
        query = input("Input your question: ").encode("utf-8")
        query = query.decode("utf-8")
    if query in ["quit", "q", "exit"]:
        sys.exit()
    result = qa_chain({"question": query, "chat_history": chat_history})
    print(result["answer"])

    chat_history.append((query, result["answer"]))
    query = None

    # Clear chat history if it exceeds a certain size to speedup
    if len(chat_history) > 10:
        chat_history = chat_history[-5:]
