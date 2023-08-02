from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
    LangchainEmbedding,
    load_index_from_storage,
    set_global_service_context,
    download_loader,
)
# from llama_hub.file.unstructured.base import UnstructuredReader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import gradio as gr
import sys
import os
import constants
import openai
import logging

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", constants.API_KEY)
openai.api_key = os.environ["OPENAI_API_KEY"]
persist_dir = os.getenv("PERSIST_DIR", "index_db")
text_embeddings_model_name = os.getenv(
    "TEXT_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002"
)
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
max_input_size = int(os.getenv("MAX_INPUT_SIZE", 4096))
num_outputs = int(os.getenv("NUM_OUTPUTS", 1024))
max_chunk_overlap = float(os.getenv("MAX_CHUNK_OVERLAP", 0.3))
chunk_size_limit = int(os.getenv("CHUNK_SIZE_LIMIT", 600))
temperature = float(os.getenv("TEMPERATURE", 0))

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def init_service_context():
    # define LLM service
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=temperature, model_name=model_name)
    )

    open_ai_embeddings = OpenAIEmbeddings(
        model=text_embeddings_model_name, chunk_size=chunk_size_limit
    )
    embeddings = LangchainEmbedding(open_ai_embeddings)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embeddings
    )
    set_global_service_context(service_context)


def load_index(directory_path):
    UnstructuredReader = download_loader('UnstructuredReader')
    documents = SimpleDirectoryReader(
        directory_path,
        file_extractor={
            ".pdf": UnstructuredReader(),
            ".docx": UnstructuredReader(),
            ".txt": UnstructuredReader(),
        },
        filename_as_id=True,
        encoding="utf-8"
    ).load_data()
    print(f"Loaded documents with {len(documents)} pages")

    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # Try to load the index from storage
        index = load_index_from_storage(storage_context)
        logging.info("Index loaded from storage.")
    except FileNotFoundError:
        # If index not found, create a new one
        logging.info("Index not found. Creating a new one...")
        index = GPTVectorStoreIndex.from_documents(documents, show_progress=True)
        # Persist index to disk
        index.storage_context.persist(persist_dir)
        logging.info("New index created and persisted to storage.")

    # Run refresh_ref_docs method to check for document updates
    refreshed_docs = index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
    print(refreshed_docs)
    print("Number of newly inserted/refreshed docs: ", sum(refreshed_docs))

    index.storage_context.persist(persist_dir)
    logging.info("Index refreshed and persisted to storage.")

    return index


def data_querying(input_text, history_text):
    index = load_index("data")
    query_engine = index.as_query_engine()
    template = """Câu hỏi cho bạn là: {text}
Bạn sẽ luôn trả lời bằng tiếng Việt và không đề cập đến tên tệp trong câu giới thiệu.
Vui lòng sử dụng thông tin từ ngữ cảnh được cung cấp để trả lời chi tiết và chính xác, dễ hiểu. Cân nhắc sử dụng các số hoặc ký hiệu liệt kê."""

    mod_question = template.format(text=input_text)
    response = query_engine.query(mod_question)

    return response.response


def run():
    load_index("data")
    gr.ChatInterface(
        fn=data_querying,
        title="MFV-GPT INFO",
    ).launch(share=False)


def main():
    if len(sys.argv) < 2:
        print("Please provide an argument. Example : build, run")
        return

    init_service_context()
    command = sys.argv[1]
    if command == "build":
        load_index("data")
    elif command == "run":
        run()
    else:
        print("Invalid command.")


if __name__ == "__main__":
    main()
