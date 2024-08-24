# LITERAL_API_KEY="lsk_5L5e055EzDfz49Zrnn4PXXvKnoZdTvqmbU197KAb8"

from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
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

def get_keyword_for_search(question):
    # Prepare the prompt for keyword extraction
    prompt = f"""
        You are a keyword extractor. 
        The user will send in their question. 
        Instead of doing a similarity search on the question itself, extract keywords from the question.
        You can add relevant key-words that are not mentioned in the question as well.
        example: What are some good dog breeds?
        keywords: [dogs, canines, german shepherds] etc.
        respond with no more than 10 keywords so that searching is quicker. Be efficient.
        Your output must be in JSON format, with a list of keywords.
    """
    
    # Retrieve the LLM from the user session
    llm = cl.user_session.get('llm')
    
    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("Question: {q}")
    ])
    
    # Create a chain with the prompt and JSON output parser
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_parser=JsonOutputParser()
    )
    
    # Extract keywords from the user's question
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
    prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", """
	        	You are a helpful AI agent. You will use the context provided to you to answer any conversation a user may have.
	        	"""),
            ("""
                    Context: {context}
            """),
	        (
	        	"""
	            	Previous conversation:\n
	            	{history}
	    		"""
	    	),
	        ("human", "{question}"),
            ("{human_input}")
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
    llm = ChatGoogleGenerativeAI(
		model = value,
		temperature = 0
		)
    
    loader = VectorDBClass()
    loader.run(
        db_type = 'FAISS',
        tokenize=False,
        create_new_db=True,
        chunk_overlap=50, chunk_size=100

    )
    cl.user_session.set('DB', loader.vectorDB)
    cl.user_session.set("llm",llm)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True, memory_key = 'history', input_key="human_input"))
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
    db = cl.user_session.get('DB')
    keywords = get_keyword_for_search(message.content)
    rel_docs = []
    print("The stream has reached here")
    print(keywords)
    for keyword in keywords:
        print(keyword)
        context = db.similarity_search(message.content)
        rel_docs.append(context)
    
    combined_input = {"question": message.content, "context":rel_docs, "human_input":""}
    print(combined_input)
    res = cl.Message(content="")
    async for chunk in runnable.astream(
        combined_input,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # print(chunk)
        await res.stream_token(chunk['text'])

    await res.send()



    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

