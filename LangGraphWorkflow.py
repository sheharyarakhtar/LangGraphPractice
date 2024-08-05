from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from google.colab import userdata
model_name = 'gemini-1.5-flash'
llm = ChatGoogleGenerativeAI(model=model_name,
                             google_api_key=userdata.get('GOOGLE_API_KEY'),
                             temperature = 0)



#----------------------------------------------------------------------------------------------------#
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(f"len of documents :{len(docs_list)}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"length of document chunks generated :{len(doc_splits)}")


vectorstore = Chroma.from_documents(documents=doc_splits,
                                    embedding=embed_model,
                                    collection_name="local-rag")

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

#----------------------------------------------------------------------------------------------------#


####################### Router Agent ########################
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
prompt = PromptTemplate(
    template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a vectorstore or web search. 
    Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. 
    You do not need to be stringent with the keywords in the question related to these topics. 
    Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
    )
start = time.time()
question_router = prompt | llm | JsonOutputParser()
question = 'llm agent memory'
print(question_router.invoke({"question":question}))
end = time.time()
print(f"The time required to generate response by router chain in seconds: {end - start}")

####################### Relevance Scoring Agent ########################
prompt = PromptTemplate(
    template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variable = ["question","document"]
)
start = time.time()
retrieval_grader = prompt | llm | JsonOutputParser()
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
end = time.time()
print(f"The time required to generate response by the retrieval grader in seconds:{end - start}")



####################### Generation Support Grading Agent ########################
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
    Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
start = time.time()
hallucination_grader = prompt | llm | JsonOutputParser()
generation = "Deep learning always requires GPUs"
hallucination_grader_response = hallucination_grader.invoke({"documents": docs, "generation": generation})
end = time.time()
print(f"The time required to generate response by the generation chain in seconds:{end - start}")
print(hallucination_grader_response)

####################### Generation Grading Agent ########################
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is useful to resolve a question. 
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
start = time.time()
answer_grader = prompt | llm | JsonOutputParser()
answer_grader_response = answer_grader.invoke({"question": question,"generation": generation})
end = time.time()
print(f"The time required to generate response by the answer grader in seconds:{end - start}")
print(answer_grader_response)

# Router Agent {question_router}
# Relevance Scoring Agent {retrieval_grader}
# Generation Support Grading Agent {hellucination_grader}
# Generation Grading Agent {answer_grader}


#--------------------------------  TOOL CONFIGURATION  ---------------------------#
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypeDict
from typing import List

os.environ['TAVILY_API_KEY'] = userdata.get('TAVILY_API_KEY')
web_search_tool = TavilySearchResults()

class GraphState(TypedDict):
	question: str
	generation: str
	web_search_tool: str
	documents: List[str]

from langchain.schema import Document

def retrieve(state):
	"""
	Retrieve documents from vectorstore
	Args:
		state (dict): The current graph state
	Returns:
		state (dict): New key added to state, docuemtns, that contain retrieved documents
	"""
	print('--Retrieve--')
	question = state['question']
	documents = retriever.invoke(question)
	return {'documents': documents, 'question':question}
def generate(state):
	"""
	Generate answer using RAG on retrieved documents
	Args:
		state (dict): Current graph state
	Returns:
		state (dict): New key added to state, generation, that contains LLM generation

	"""
	question = state['question']
	documents = state['documents']
	generation = rag_chain.invoke({'context':documents, 'question':question})
	return {"documents":documents, "question": question, "generation": generation}

def grade_documents(state):
	"""
	Determine whether the retrieved documents are relevant to the question
	If any document is not relevant, we will set a flag to run web search
	Args:
		state (dict): Current graph state
	Returns:
		state (dict): Filtered out irrelevant documents and updated web_search state
	"""
	print("----Checking documents relevance to question ----")
	question = state['question']
	documents = state['documents']
	filtered_docs = []
	web_search = 'No'
	for d in documents:
		score = retrieval_grader.invoke({'question':question, 'documents': d.page_content})
		grade = score['score']
		if grade.lower() == 'yes':
			print("--- Grade: Document relevant ----")
			filtered_docs.append(d)
		else:
			print("--- Grade: Document not relevant ---")
			web_search = 'Yes'
			continue
	return {'documents': filtered_docs, 'question':question, 'web_search': web_search}
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

#--------------------------------  Agent Configuration  ---------------------------#
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
	print("-- Route question --")
	question = state['question']
	print(question)
	source = question_router.invoke({'question':question})
	print(source)
	print(source['datasource'])
	if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
	question = state['question']
	web_search = state['web_search']
	filtered_docs = state['documents']
	if web_search == 'Yes':
		print("----- Decision: All documents are not relevant. Search the web. ---")
		return 'websearch'
	else:
		print("--- Decision: Generate ---")
		return 'generate'

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# ---------------------- Orchestration ------------------------#
from langgraph.graph import END, StateGraph
workflow = StateGraph(GraphState)
workflow.add_node('websearch', web_search)
workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node('generate', generate)



workflow.add_conditional_entry_point(
	route_question,
	{
		'websearch':'websearch',
		'vectorstore':'retrieve'
	},
	)

workflow.add_edge('retrieve','grade_documents')
workflow.add_conditional_edge(
	'grade_documents',
	decide_to_generate,
	{
		'websearch':'websearch',
		'generate':'generate'
	},
	)
workflow.add_edge('websearch','generate')
workflow.add_conditional_edge(
	'generate',
	grade_generation_v_documents_and_question,
	{
		'not supported': 'generate',
		'useful': END,
		'not useful': 'websearch'
	}
	)
