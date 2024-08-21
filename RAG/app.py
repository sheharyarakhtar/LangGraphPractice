# LITERAL_API_KEY="lsk_5L5e055EzDfz49Zrnn4PXXvKnoZdTvqmbU197KAb8"

from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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



def setup_runnable():
	llm = ChatGoogleGenerativeAI(
		model = cl.user_session.get('model_name'),
		temperature = 0
		)

	memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
	prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", """
	        	You are a helpful AI agent.
	        	"""),
	        (
	        	"""
	            	Previous conversation:\n
	            	{history}
	    		"""
	    	),
	        ("human", "{question}"),
	    ]
	)

	runnable = LLMChain(llm = llm, prompt = prompt, verbose = True, memory = memory) 
    # (
    #     RunnablePassthrough.assign(
    #         history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    #     )
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )
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
    cl.user_session.set("model_name",value)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    # memory = cl.user_session.get('memory')
    print("Memory:", memory)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    print(root_messages)
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk['text'])

    await res.send()



    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

