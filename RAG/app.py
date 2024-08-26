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
    """
    Extracts keywords from a given question using a language model (LLM).
    
    Args:
        question (str): The user's question or query.
    
    Returns:
        list: A list of extracted keywords, or an empty list if extraction fails.
    """
    llm = cl.user_session.get('llm')  # Retrieve the language model from the user session.
    chain = LLMChain(
        llm=llm,
        prompt=search_prompt,
        output_parser=JsonOutputParser()
    )
    try:
        # Invoke the chain and parse the JSON output to get the keywords.
        queries = chain.invoke({"q": question})
        print(queries)
        
        # Extract the 'text' field from the JSON output as the keywords.
        keywords = queries['text']
        print(keywords)
        return keywords
    except Exception as e:
        # Handle any errors that occur during keyword extraction.
        print(f"Error extracting keywords: {e}")
        return []

def setup_runnable():
    """
    Sets up the main LLM chain to be used during the chat session.
    
    This function initializes the LLMChain with the current language model, 
    the RAG prompt, and conversation memory, then stores it in the user session.
    """
    llm = cl.user_session.get('llm')  # Retrieve the language model from the user session.
    memory = cl.user_session.get("memory")  # Retrieve conversation memory from the user session.

    # Initialize the LLM chain with the provided prompt and memory.
    runnable = LLMChain(llm=llm, prompt=rag_prompt, verbose=True, memory=memory)
    
    # Store the runnable chain in the user session for future use.
    cl.user_session.set("runnable", runnable)

@cl.password_auth_callback
def auth():
    """
    Authenticates the user by returning a user object.
    
    Returns:
        cl.User: A User object with a test identifier.
    """
    return cl.User(identifier="test")

from chainlit.input_widget import Select

@cl.on_chat_start
async def on_chat_start():
    """
    Handles the initialization process when a new chat session starts.
    
    This function prompts the user to select a model, initializes the vector 
    database, and sets up the necessary components for the chat session.
    """
    # Prompt the user to select a model from a list of options.
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
    
    # Retrieve the selected model and initialize the LLM.
    value = settings["Model"]
    llm = ChatGoogleGenerativeAI(
        model=value,
        temperature=0,
        stream=True
    )
    
    # Initialize the vector database class and run it.
    loader = VectorDBClass()
<<<<<<< Updated upstream
    loader.run(
        db_type='FAISS',
=======
    loader.setup(
        db_type = 'FAISS',
>>>>>>> Stashed changes
        tokenize=False,
        create_new_db=False,
        chunk_overlap=50, chunk_size=250
    )
    
    # Store the vector database, LLM, and memory in the user session.
    cl.user_session.set('DB', loader.vectorDB)
    cl.user_session.set("llm", llm)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True, memory_key='history', input_key="human_input"))
    
    # Set up the runnable LLM chain.
    setup_runnable()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Handles the process when a chat session is resumed.
    
    This function reinitializes the chat settings, restores conversation memory, 
    and sets up the necessary components for continuing the chat session.
    
    Args:
        thread (ThreadDict): A dictionary containing the previous chat messages.
    """
    # Prompt the user to select a model from a list of options.
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
    
    # Retrieve the selected model and initialize the LLM.
    value = settings["Model"]
    llm = ChatGoogleGenerativeAI(
        model=value,
        temperature=0,
        stream=True
    )
    
    # Initialize the vector database class and run it.
    loader = VectorDBClass()
<<<<<<< Updated upstream
    loader.run(
        db_type='FAISS',
=======
    loader.setup(
        db_type = 'FAISS',
>>>>>>> Stashed changes
        tokenize=False,
        create_new_db=False,
        chunk_overlap=50, chunk_size=250
    )
    
    # Store the vector database and LLM in the user session.
    cl.user_session.set('DB', loader.vectorDB)
    cl.user_session.set("llm", llm)
    
    # Rebuild the conversation memory from the previous chat messages.
    memory = ConversationBufferMemory(return_messages=True, memory_key='history', input_key="human_input")
    root_messages = [m for m in thread["steps"] if (m["parentId"] == None) or (m['name'] == 'Assistant')]
    for message in root_messages:
        print(message)
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])
    print(memory)
    
    # Store the restored memory in the user session and set up the runnable chain.
    cl.user_session.set("memory", memory)
    setup_runnable()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user during the chat session.
    
    This function processes the message, extracts relevant keywords, 
    performs a similarity search in the vector database, and generates 
    a response using the LLM chain.
    
    Args:
        message (cl.Message): The incoming message from the user.
    """
    memory = cl.user_session.get("memory")  # Retrieve conversation memory from the user session.
    runnable = cl.user_session.get("runnable")  # Retrieve the runnable LLM chain from the user session.
    db = cl.user_session.get('DB')  # Retrieve the vector database from the user session.
    
    # Extract keywords from the user's message.
    keywords = get_keyword_for_search(message.content)
    rel_docs = []
    
    print("The stream has reached here")
    print(keywords)
    
    # Perform a similarity search in the vector database for each keyword.
    for keyword in keywords:
        context = db.similarity_search(keyword, k = 1)
        print(keyword, context)
        rel_docs.append(context)
    
    # Combine the user's question, retrieved context, and empty human input.
    combined_input = {
        "question": message.content, 
        "context": rel_docs, 
        "human_input": ""
    }
    
    # Prepare to send the response as a stream.
    res = cl.Message(content="")
    async for chunk in runnable.astream(
        combined_input,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        # Stream the response token by token.
        await res.stream_token(chunk['text'])

    # Send the final response message.
    await res.send()
<<<<<<< Updated upstream


    # memory.chat_memory.add_user_message(message.content)
    # memory.chat_memory.add_ai_message(res.content)

=======
>>>>>>> Stashed changes
