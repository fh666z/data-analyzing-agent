import numpy as np
import pandas as pd
import matplotlib
import seaborn
import sklearn
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
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

# System prompt for the agent
system_prompt = (
    "You are a data science assistant. Use the available tools to analyze CSV files. "
    "Your job is to determine whether each dataset is for classification or regression, based on its structure."
)

tools = [list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset]

# Create the agent using LangGraph's prebuilt create_react_agent
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

print("📊 Ask questions about your dataset (type 'exit' to quit):")
while True:
    user_input = input(" You: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("see ya later")
        break
    
    # Invoke the agent with the new LangGraph format
    result = agent.invoke({"messages": [("user", user_input)]})
    
    # Extract the last message (the agent's response)
    last_message = result["messages"][-1]
    content = last_message.content
    
    # Handle Gemini's structured response format
    if isinstance(content, list):
        # Extract text from the content blocks
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        output = "".join(text_parts)
    else:
        output = content
    
    print(f"my Agent: {output}")