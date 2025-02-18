import os
import subprocess
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
 
 
# --- Existing Tool: Wikipedia Search ---
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
 
 
@tool
def wikipedia_search(query: str) -> str:
   '''Return a Wikipedia page containing information about the given query.'''
   if query:
       return f"Here's the information about {query}: {wikipedia.run(query)}"
   else:
       return "No query received as Argument."
 
 
# --- Environment Setup ---
os.environ["TAVILY_API_KEY"] = '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'
 
 
# --- Existing Tool: Web Search ---
from langchain_community.tools import TavilySearchResults
web_search_tool = TavilySearchResults(
   max_results=3,
   search_depth="advanced",
   include_answer=True,
   include_raw_content=True,
   description='''A search engine optimized for comprehensive, accurate, and trusted results.
   Useful for when you need to answer questions about current events. Input should be a search query.'''
)
 
 
# --- Existing Tool: Python REPL for Plotting ---
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()
 
 
from typing import Annotated
 
 
@tool
def python_repl_tool(
   code: Annotated[str, "The python code to execute to generate your plot."]
):
   """Executes the given Python code to generate a plot and save the output."""
   try:
       result = repl.run(code)
   except BaseException as e:
       return f"Failed to execute. Error: {repr(e)}"
   result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
   return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
 
 
# --- New Tool: Launch Streamlit App ---
@tool
def launch_streamlit_app(dummy_input: str) -> str:
   """
   Launches the Streamlit app (app.py) automatically.
   The dummy_input parameter is ignored but is required to match the expected tool signature.
   """
   try:
       subprocess.Popen(["streamlit", "run", "app.py"])
       return "Streamlit app launched successfully."
   except Exception as e:
       return f"Failed to launch Streamlit app: {str(e)}"
 
 
# --- Tools List ---
tools = [web_search_tool, python_repl_tool, launch_streamlit_app]
 
 
# --- LLM Setup ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
 
 
llm = HuggingFaceEndpoint(
   repo_id="mistralai/Mistral-Nemo-Instruct-2407",
   task='text-generation',
   huggingfacehub_api_token='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',
   max_new_tokens=512,
   temperature=0.8,
   verbose=True
)
model = ChatHuggingFace(verbose=True, llm=llm)
 
 
# --- Agent Template ---
template = '''You are an expert research scientist and data analyst.
Answer questions as accurately as possible using the available tools:
 
 
1] tavily_search_results_json – Use this tool only when external data is required.
2] python_repl_tool – Use this tool only for executing Python plots. Input should be python code.
3] launch_streamlit_app – Use this tool to automatically launch the Streamlit app that displays the output.
 
 
Strict Rules:
- DO NOT use tools unless necessary.
- DO NOT use `python_repl_tool` for basic text responses.
- Hardcode data values in Python code instead of dynamically referencing variables.
- Ensure Python code is syntactically correct and formatted properly.
- ALWAYS WAIT FOR OBSERVATION TO RETURN AFTER ACTION INPUT.
 
 
Example Workflow:
Question: Plot India’s population progression for the last 3 decades.
Thought: I need India’s population data. I should search for this information.
Action: tavily_search_results_json
Action Input: India's population data for the last 3 decades
 
 
(STOP. Wait for observation to return)
 
 
Observation: India's population data is: [data here].
 
 
Thought: I now have the data. I will plot this using Python.
Action: python_repl_tool
Action Input:
 
 
import os
import matplotlib.pyplot as plt
# Ensure the output directory exists
os.makedirs("output", exist_ok=True)
years = [1990, 2000, 2010, 2020, 2023]
population = [0.873, 1.05, 1.23, 1.38, 1.42]
plt.figure(figsize=(8,5))
plt.plot(years, population, marker='o', linestyle='-', color='blue', label="Population (Billions)")
plt.xlabel("Year")
plt.ylabel("Population (Billions)")
plt.title("India's Population Growth (1990-2023)")
plt.legend()
plt.grid(True)
plt.savefig("output/plot.png")
plt.close()
 
 
(STOP. Wait for observation to return)
 
 
Observation: Plot saved successfully.
 
 
Thought: Now that I have saved the plot, I will launch the Streamlit app to display the result.
Action: launch_streamlit_app
Action Input: dummy_data
 
 
(STOP. Wait for observation to return)
 
 
Observation: Streamlit app launched successfully.
Final Answer: The graph has been plotted and the Streamlit app is now running to display the output.
 
 
Now, respond to the user's question while following these instructions.
 
 
Question: {input}
'''
 
 
# --- Initialize and Invoke the Agent ---
from langchain.agents import initialize_agent
agent = initialize_agent(tools, llm, verbose=True, handle_parsing_errors=True)
user_question = "Plot UK's population growth for the past few years."
formatted_prompt = template.format(input=user_question)
 
 
agent.invoke(formatted_prompt)
