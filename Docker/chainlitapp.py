import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
# from IPython.display import display, Markdown
import os
# from google.colab import userdata
# import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph
import yaml

llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                             api_key = os.environ.get('GOOGLE_API_KEY'),
                             temperature = 0,
                             verbose = True)
web_search_tool = TavilySearchResults()
prompt = PromptTemplate(
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_ud|>
    You are a web researcher. Based on the question provided to you,
    you will return 3 search queries that can be searched online to find the answer to the question asked.
    Provide you answer as a JSON with no preamble or explanation. No need to add anything, just the JSON.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the question:
    {question}""",
    input_variables = ['question'])
query_former = prompt | llm | JsonOutputParser()
prompt = PromptTemplate(
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_ud|>
    You are a writer agent. You will be provided with the results of a websearch along with the question you need to answer.
    Based on the contents of the search result and the question,
    You must provide some text which will answer the user's quesiton along with reference links directing users to read the full article if they want.
    You must give citations to the articles and give a 10 line summary.
    Give as much information as possible.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the search results:
    \n ----- \n
    {search_results}
    \n ----- \n
    Here is the question:
    {question}""",
    input_variables = ['search_results','question']
)
summarizer = prompt | llm | StrOutputParser()
prompt = PromptTemplate(
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_ud|>
    You are a validating agent. You will be provided with some documents, a question and an answer.
    You must verify whether the question asked by the user has been properly answered or not.
    The answer must include: a paragraph based on the documents provided AND attached links for referece of the user.
    Both must be present for the answer to qualify, else provide feedback.
    Your output must be in a JSON format with two keys:
    1- 'validation', which can have a value of yes or no (binary)
    and 'feedback' which will be populated if the validation is 'no'. In which case, you must provide feedback on why the answer is no
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is facts from the document:
    \n ----- \n
    {search_results}
    \n ----- \n
    Here is the question:
    {question}
    \n ----- \n
    Here is the answer:
    {generation}
    """,
    input_variables = ['search_results','question']
)
validator = prompt | llm | JsonOutputParser()
class GraphState(TypedDict):
  question: str
  keywords: List[str]
  search_results: str
  validator: str
  generation: str

def keyword_retriever(state):
  question = state['question']
  keywords = query_former.invoke({'question':question})
  # print(keywords)
  return {'keywords':keywords}

def search_web(state):
  documents = []
  keywords = state['keywords']
  for search_query in keywords:
    docs = web_search_tool(search_query)
    web_results = "\n".join(d['content'] + "\nReference: "+d['url'] for d in docs)
    web_results = Document(page_content = web_results)
    if documents is not None:
      documents.append(web_results)
    else:
      documents = [web_results]
    # print(documents)
  return {'search_results':documents}

def generate(state):
  question = state['question']
  search_results = state['search_results']
  generation = summarizer.invoke({'search_results':search_results, 'question':question})
  # print(generation)
  return {'generation':generation}

def validate(state):
  question = state['question']
  search_results = state['search_results']
  generation = state['generation']
  validate = validator.invoke({'question':question,
                    'search_results':search_results,
                    'generation':generation})
  if validate['validation'] == 'yes':
    return {'generation':generation}




# query_former -> websearch -> summarizer -> validator
workflow = StateGraph(GraphState)
workflow.add_node('keyword_retriever', keyword_retriever)
workflow.set_entry_point('keyword_retriever')
workflow.add_node('search_web', search_web)
workflow.add_node('generate', generate)
workflow.add_node('validate', validate)

workflow.add_edge('keyword_retriever','search_web')
workflow.add_edge('search_web','generate')
workflow.add_edge('generate','validate')
app = workflow.compile()


def get_result(input: str) -> str:
  input = {'question':input}
  results = []
  for i in app.stream(input):
   result = list(i.values())[0]
   print(result)
   results.append(result)
  
  ret = results[-1]
  return ret['generation']


@cl.on_message
async def main(message: cl.Message):
   await cl.Message(
        content=get_result(message.content),
    ).send()
