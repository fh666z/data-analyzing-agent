import numpy as np
import pandas as pd
import matplotlib
import seaborn
import sklearn
import langchain
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils import list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method
from utils_evaluation import evaluate_classification_dataset, evaluate_regression_dataset



# Use the selected Google Gemini model
# Note: Using Gemini 2.5 instead of 3.x because Gemini 3 requires thought signatures
# for function calling, which LangGraph doesn't automatically handle yet.
# See: https://ai.google.dev/gemini-api/docs/thought-signatures
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Optional: can also be auto-detected from env
)

# 🧠 Step 2: Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a data science assistant. Use the available tools to analyze CSV files. "
     "Your job is to determine whether each dataset is for classification or regression, based on its structure."),
    
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Required for tool-calling agents
])

tools=[list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset]


agent = create_tool_calling_agent(llm, prompt, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
agent_executor.agent.stream_runnable = False

print("📊 Ask questions about your dataset (type 'exit' to quit):")

while True:
    user_input=input(" You:")
    if user_input.strip().lower() in ['exit','quit']:
        print("see ya later")
        break
        
    result=agent_executor.invoke({"input":user_input})
    print(f"my Agent: {result['output']}")