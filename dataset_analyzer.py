import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from utils import DatasetManager, create_dataset_tools
from utils_evaluation import create_evaluation_tools


# Create a single DatasetManager instance for this session
dataset_manager = DatasetManager(max_datasets=10)

# Create tools bound to the manager
dataset_tools = create_dataset_tools(dataset_manager)
evaluation_tools = create_evaluation_tools(dataset_manager)

# Use the selected Google Gemini model
# Note: Using Gemini 2.5 instead of 3.x because Gemini 3 requires thought signatures
# for function calling, which LangGraph doesn't automatically handle yet.
# See: https://ai.google.dev/gemini-api/docs/thought-signatures
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# System prompt for the agent
system_prompt = (
    "You are a data science assistant. Use the available tools to analyze CSV files. "
    "Your job is to determine whether each dataset is for classification or regression, based on its structure. "
    "If it is something else, try to suggest what the dataset represents."
)

# Combine all tools into a single list
tools = [
    # Reading/Analysis tools
    dataset_tools["list_csv_files"],
    dataset_tools["preload_datasets"],
    dataset_tools["get_dataset_summaries"],
    dataset_tools["call_dataframe_method"],
    # Editing tools
    dataset_tools["drop_column"],
    dataset_tools["rename_column"],
    dataset_tools["drop_rows_with_missing"],
    dataset_tools["fill_missing_values"],
    dataset_tools["filter_dataset"],
    dataset_tools["save_dataset"],
    # Evaluation tools
    evaluation_tools["evaluate_classification_dataset"],
    evaluation_tools["evaluate_regression_dataset"],
]

# Create the agent
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def extract_response_text(content) -> str:
    """Extract text from Gemini's structured response format."""
    if isinstance(content, list):
        text_parts = [
            block.get("text", "") 
            for block in content 
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts)
    return content


def main():
    print("📊 Ask questions about your dataset (type 'exit' to quit):")
    
    while True:
        user_input = input(" You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("see ya later")
            break
        
        # Invoke the agent
        result = agent.invoke({"messages": [("user", user_input)]})
        
        # Extract and display the response
        last_message = result["messages"][-1]
        output = extract_response_text(last_message.content)
        print(f"my Agent: {output}")


if __name__ == "__main__":
    main()
