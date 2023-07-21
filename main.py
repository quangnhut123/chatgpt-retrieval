from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import constants
import openai

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", constants.API_KEY)
openai.api_key = os.environ["OPENAI_API_KEY"]
persist_dir = os.getenv("PERSIST_DIR", "index_db")
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
max_input_size = int(os.getenv("MAX_INPUT_SIZE", 4096))
num_outputs = int(os.getenv("NUM_OUTPUTS", 1024))
max_chunk_overlap = float(os.getenv("MAX_CHUNK_OVERLAP", 0.2))
chunk_size_limit = int(os.getenv("CHUNK_SIZE_LIMIT", 600))
temperature = float(os.getenv("TEMPERATURE", 0))


def init_service_context():
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=num_outputs,
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return service_context


def data_indexing(directory_path, service_context):
    documents = SimpleDirectoryReader(
        directory_path, encoding="utf-8", recursive=True
    ).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )
    index.storage_context.persist(persist_dir)

    return index


def data_querying(input_text, service_context):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine()
    template = """
        You are an AI language model designed to provide helpful answers based on provided context.
        You will answer in Vietnamese language and do not mention about filename in introduce sentence.
        Mention a part of question as introduce sentence.
        Please use the information from the provided context to answer accurately.
        Ensure your answers in details with clear context for easy understanding. Consider using listing numbers or symbols.
        If the provided context contains a Kibela link then include kibela link at the end of your answer with a two-line break.
        If there is no Kibela link in the context, do not include it in your answer.
        The question is: {text}
    """

    mod_question = template.format(text=input_text)
    response = query_engine.query(mod_question)

    return response.response


def run(service_context):
    gr.Interface(
        fn=lambda x: data_querying(x, service_context),
        inputs=gr.components.Textbox(lines=7, label="Enter your question here"),
        outputs=gr.components.Textbox(lines=7, label="Your answer"),
        title="MFV GPT Trained By Private Data",
    ).launch(share=False)


def main():
    if len(sys.argv) < 2:
        print("Please provide an argument. Example : build, run")
        return

    service_context = init_service_context()
    command = sys.argv[1]
    if command == "build":
        data_indexing("data", service_context)
    elif command == "run":
        run(service_context)
    else:
        print("Invalid command.")


if __name__ == "__main__":
    main()
