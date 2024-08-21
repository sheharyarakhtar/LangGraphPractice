from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
import getpass
import os
# from langchain.llms import OpenAI
from IPython.display import display, Markdown
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from neo4j import GraphDatabase
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_experimental.graph_transformers.llm import *
from langchain_community.tools.tavily_search import TavilySearchResults
import yaml

with open('config.yaml','r') as file:
  config = yaml.safe_load(file)

os.environ['TAVILY_API_KEY'] = config['TAVILY_API_KEY']
google = os.environ['GOOGLE_API_KEY'] = config['GOOGLE_API_KEY']
model_name = 'gemini-1.5-flash'

llm = ChatGoogleGenerativeAI(api_key = google, model = model_name, temperature = 0)
web_search_tool = TavilySearchResults()
prompt = PromptTemplate(
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_ud|>
    You are an elaborater agent. You will be provided with the results of a websearch.
    Based on the contents of the search result,
    You must write an extensive report with reference links directing users to read the full article if they want.
    Give as much information as possible.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the search results:
    \n ----- \n
    {search_results}
    """,
    input_variables = ['search_results','question']
)
summarizer = prompt | llm | StrOutputParser()
queries = ['Bitcoin news August 2024','Ethereum news August 2024','Pendle price prediction August 2024']
documents = []
for query in queries:
  result = web_search_tool(query)
#  web_results = "\n".join(d['content'] + "\nReference: "+d['url'] for d in result)
  web_results = result
  web_results = Document(page_content = web_results)
  documents.append(web_results)
print("Running Summarizer")
text = summarizer.invoke({'search_results':documents})
print(text)


