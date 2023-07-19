from llama_index import SimpleDirectoryReader,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import constants
import openai

os.environ["OPENAI_API_KEY"] = constants.API_KEY
openai.api_key = constants.API_KEY


def construct_index(directory_path):
    max_input_size = 8192
    num_outputs = 4000
    max_chunk_overlap = 0.2
    chunk_size_limit = 1000
    model_name = "gpt-4"
    temperature = 0.2

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=temperature, model_name=model_name, max_tokens=num_outputs
        )
    )
    documents = SimpleDirectoryReader(
        directory_path, encoding="utf-8", recursive=True
    ).load_data()
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    index = GPTVectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        prompt_helper=prompt_helper,
        show_progress=True,
    )
    # index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.set_index_id("vector_index")
    index.storage_context.persist(persist_dir="index_db")

    return index


def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="index_db")
    index = load_index_from_storage(storage_context, index_id="vector_index")
    query_engine = index.as_query_engine()
    template = """
    You are act as slack chatbot. You should answer with the same language as the question, try to make answer clear in details and easy to understand by using listing number or symbols style. Extract kibela link from source data and put at the end of answer with 2 lines break as following format "***Please read more: kibela_link_extracted" but translate as language of given question. The question is: {text}
    """
    mod_question = template.format(text=input_text)
    response = query_engine.query(mod_question)
    return response.response


def run():
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.components.Textbox(lines=7, label="Enter your question here"),
        outputs=gr.components.Textbox(lines=7, label="Your answer"),
        title="MFV GPT Trained By Private Data",
    ).launch(share=False)

def main():
    if len(sys.argv) < 2:
        print("Please provide an argument. Example : build, run")
        return

    command = sys.argv[1]
    if command == "build":
        construct_index("data")
    elif command == "run":
        run()
    else:
        print("Invalid command.")

if __name__ == "__main__":
    main()
