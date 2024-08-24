from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from prompts import *
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from chainlit.types import ThreadDict
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
import chainlit as cl
from vectorDB import VectorDBClass
from langchain_core.output_parsers import JsonOutputParser
from literalai import LiteralClient

def get_keyword_for_search(question):
    llm = cl.user_session.get('llm')
    chain = LLMChain(
        llm=llm,
        prompt=search_prompt,
        output_parser=JsonOutputParser()
    )
    try:
        # Invoke the chain and parse the JSON output
        queries = chain.invoke({"q":question})
        print(queries)
        
        # Extract keywords from the JSON result
        keywords = queries['text']
        print(keywords)
        return keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []
    



def setup_runnable():
    llm = cl.user_session.get('llm')
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = LLMChain(llm = llm, prompt = rag_prompt, verbose = True, memory = memory)
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")

from chainlit.input_widget import Select


@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Gemini Models",
                values=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]
    llm = ChatGoogleGenerativeAI(
		model = value,
		temperature = 0
		)
    loader = VectorDBClass()
    loader.run(
        db_type = 'FAISS',
        tokenize=False,
        create_new_db=False,
        chunk_overlap=50, chunk_size=250
    )
    cl.user_session.set('DB', loader.vectorDB)
    cl.user_session.set("llm",llm)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True, memory_key = 'history', input_key="human_input"))
    setup_runnable()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Gemini Models",
                values=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]
    llm = ChatGoogleGenerativeAI(
		model = value,
		temperature = 0
		)
    loader = VectorDBClass()
    loader.run(
        db_type = 'FAISS',
        tokenize=False,
        create_new_db=False,
        chunk_overlap=50, chunk_size=250
    )
    cl.user_session.set('DB', loader.vectorDB)
    cl.user_session.set("llm",llm)
    memory = ConversationBufferMemory(return_messages=True, memory_key = 'history', input_key="human_input")
    root_messages = [m for m in thread["steps"] if (m["parentId"] == None) or (m['name']=='Assistant')]
    for message in root_messages:
        print(message)
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])
    print(memory)
    cl.user_session.set("memory", memory)
    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable
    db = cl.user_session.get('DB')
    keywords = get_keyword_for_search(message.content)
    rel_docs = []
    print("The stream has reached here")
    print(keywords)
    for keyword in keywords:
        context = db.similarity_search(message.content)
        rel_docs.append(context)
    
    combined_input = {"question": message.content, 
                      "context":rel_docs, 
                      "human_input":""}
    res = cl.Message(content="")
    async for chunk in runnable.astream(
        combined_input,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        await res.stream_token(chunk['text'])

    await res.send()



    # memory.chat_memory.add_user_message(message.content)
    # memory.chat_memory.add_ai_message(res.content)

