from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC7xE1m66Pv_lq1MNLZMbhxbzOKZf-4mPU'


@cl.on_chat_start
async def on_chat_start():
    model = ChatGoogleGenerativeAI(
    	api_key = os.environ.get('GOOGLE_API_KEY'),
    	temperature = 0,
    	model = 'gemini-1.5-flash',
    	verbose = False
    	)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()