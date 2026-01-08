from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain import hub
from backend.groq import GROQ_CHAT

# Initialize Groq LLM from connection pool
llm = GROQ_CHAT.get_client(model="mixtral-8x7b-32768", temperature=0)

# Define tools
@tool
def search_food_info(query: str) -> str:
    """Search for food information"""
    return f"Food info about {query}"

@tool
def get_nutrition_data(food: str) -> str:
    """Get nutrition data for a food item"""
    return f"Nutrition data for {food}"

# Create agent
tools = [search_food_info, get_nutrition_data]
prompt = hub.pull("hwchase17/tool-calling-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
if __name__ == "__main__":
    result = agent_executor.invoke({"input": "What are the nutrients in an apple?"})
    print(result["output"])