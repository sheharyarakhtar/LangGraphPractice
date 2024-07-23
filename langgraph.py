# -*- coding: utf-8 -*-
"""langgraph.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uBW2T1N2TVrVfUY4nHl0iFkEjOIwQ5Xc
"""

# !pip install -U langgraph
# %pip install langchain_anthropic
# %pip install langchain_google_genai
# %pip install langchain_community
# %pip -q install langchain_experimental
# %pip -q install streamlit
# %pip -q install langchainhub

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, Markdown

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."
tools = [search]
tool_node = ToolNode(tools)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                               verbose=True,
                               google_api_key=userdata.get('GOOGLE_API_KEY'),
                               temperature = 0)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": "what is the weather in sf?"},
    config={"configurable": {"thread_id": 42}}
)
display(Markdown(final_state["messages"][-1].content))

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import os
from langchain_experimental.utilities import PythonREPL
os.environ['TAVILY_API_KEY'] = userdata.get('TAVILY_API_KEY')
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

import operator
from typing import Annotated, Sequence, TypedDict


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

import functools
from langchain_core.messages import AIMessage

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                               verbose=True,
                               google_api_key=userdata.get('GOOGLE_API_KEY'),
                               temperature = 0)
# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

from langgraph.prebuilt import ToolNode

tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)



from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentActionMessageLog
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="LangChain Agent", layout="wide")
def main():
  st.title('LangGraph Agent + Gemini + Custom Tool + Streamlit')
  input_text = st.text_area('Enter your text:')
  if st.button('Run Agent'):
    def search(query: str):
        """Call to surf the web."""
        return DuckDuckGoSearchRun().run(query)

    def toggle_case(word):
                toggled_word = ""
                for char in word:
                    if char.islower():
                        toggled_word += char.upper()
                    elif char.isupper():
                        toggled_word += char.lower()
                    else:
                        toggled_word += char
                return toggled_word

    def sort_string(string):
        return ''.join(sorted(string))

    tools = [
          Tool(
              name = "Search",
              func=search,
              description="useful for when you need to answer questions about current events",
          ),
          Tool(
              name = "Toogle_Case",
              func = lambda word: toggle_case(word),
              description = "use when you want covert the letter to uppercase or lowercase",
          ),
          Tool(
              name = "Sort String",
              func = lambda string: sort_string(string),
              description = "use when you want sort a string alphabetically",
          ),

            ]
    from langchain import hub
    prompt = hub.pull("hwchase17/react")

    llm = ChatGoogleGenerativeAI(
        model = 'gemini-1.5-flash',
        google_api_key = userdata.get('GOOGLE_API_KEY'),
        temperature = 0,
        verbose = True
    )
    agent_runnable = create_react_agent(llm, tools, prompt)
    class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage]
        agent_outcome: Union[AgentAction, AgentFinish, None]
        return_direct: bool
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    tool_executor = ToolExecutor(tools)

    def run_agent(state):
        """
        #if you want to better manages intermediate steps
        inputs = state.copy()
        if len(inputs['intermediate_steps']) > 5:
            inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
        """
        agent_outcome = agent_runnable.invoke(state)
        return {"agent_outcome": agent_outcome}
    def execute_tools(state):

        messages = [state['agent_outcome'] ]
        last_message = messages[-1]
        ######### human in the loop ###########
        # human input y/n
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        # state_action = state['agent_outcome']
        # human_key = input(f"[y/n] continue with: {state_action}?")
        # if human_key == "n":
        #     raise ValueError

        tool_name = last_message.tool
        arguments = last_message
        if tool_name == "Search" or tool_name == "Sort" or tool_name == "Toggle_Case":

            if "return_direct" in arguments:
                del arguments["return_direct"]
        action = ToolInvocation(
            tool=tool_name,
            tool_input= last_message.tool_input,
        )
        response = tool_executor.invoke(action)
        return {"intermediate_steps": [(state['agent_outcome'],response)]}
    def should_continue(state):

                messages = [state['agent_outcome'] ]
                last_message = messages[-1]
                if "Action" not in last_message.log:
                    return "end"
                else:
                    arguments = state["return_direct"]
                    if arguments is True:
                        return "final"
                    else:
                        return "continue"
    def first_agent(inputs):
                action = AgentActionMessageLog(
                tool="Search",
                tool_input=inputs["input"],
                log="",
                message_log=[]
                )
                return {"agent_outcome": action}
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)
    workflow.add_node("final", execute_tools)
    # uncomment if you want to always calls a certain tool first
    # workflow.add_node("first_agent", first_agent)


    workflow.set_entry_point("agent")
    # uncomment if you want to always calls a certain tool first
    # workflow.set_entry_point("first_agent")

    workflow.add_conditional_edges(

        "agent",
        should_continue,

        {
            "continue": "action",
            "final": "final",
            "end": END
        }
    )


    workflow.add_edge('action', 'agent')
    workflow.add_edge('final', END)
    # uncomment if you want to always calls a certain tool first
    # workflow.add_edge('first_agent', 'action')
    app = workflow.compile()
    inputs = {"input": input_text, "chat_history": [], "return_direct": False}
    results = []
    for s in app.stream(inputs):
        result = list(s.values())[0]
        results.append(result)
        st.write(result)  # Display each step's output

main()















