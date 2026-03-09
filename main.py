from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from middleware import EnergyMiddleware, get_model_costs
from reporting import get_total_energy_usage, PAST_24H_ENERGY_J, ENERGY_BUDGET_J

# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

@tool("get_weather", description="Get the weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is rainy and 21.7 degrees celcius"

SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant. You are an expert at researching the weather. Respond in a whimsical tone.
You have the following tools:
- get_weather: this tool takes a city as input and returns the weather in that city.
"""

tracker = EnergyMiddleware()

def create_weather_subagent(model_name: str):
    return create_agent(
        model=ChatOllama(model=model_name),
        tools=[get_weather],
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[tracker],
    )

MODEL_TIERS = {
    "large": "qwen3.5:9b",
    "medium": "qwen3.5:4b",
    "small": "qwen3.5:2b"
}

weather_agents = {
    tier: create_weather_subagent(model)
    for tier, model in MODEL_TIERS.items()
}


@tool("get_model_energy_costs", description="Returns energy consumption per input and output token for available models")
def get_model_energy_costs() -> str:
    costs = get_model_costs()
    lines = []

    for tool_name, model_name in MODEL_TIERS.items():
        model_cost = costs.get(model_name)
        lines.append(
            f"- {tool_name} ({model_name}): "
            f"{model_cost['input_token_energy']} J per input token, "
            f"{model_cost['output_token_energy']} J per output token."
        )

    return "\n".join(lines)

@tool("get_energy_usage_last_24h", description="Returns total energy consumption in joules in the past 24 hours")
def get_energy_usage_last_24h() -> str:
    return f"Energy used in the past 24 hours: {PAST_24H_ENERGY_J} joules. Budget: {ENERGY_BUDGET_J} joules."

def make_weather_tool(tier, agent):
    @tool(f"weather_{tier}", description=f"Use the {tier} accuracy weather research model")
    def call_agent(query: str) -> str:
        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].content

    return call_agent

weather_tools = [
    make_weather_tool(tier, agent)
    for tier, agent in weather_agents.items()
]


MAIN_SYSTEM_PROMPT = """
You are an assistant that answers user questions while managing energy consumption.

Available models:
- weather_large
- weather_medium
- weather_small

You can use the following tools:

get_energy_usage_last_24h
    Returns the energy used in the past 24 hours and the allowed energy budget.

get_model_energy_costs
    Returns the energy cost per input and output token for each model.

Decision strategy:

1. First check the energy usage in the past 24 hours.
2. Inspect the model energy costs.
3. Estimate how many tokens the response may require.
4. Choose the model that balances accuracy and the energy budget.

You CANNOT exceed the energy budget. Always choose a model that keeps you within the budget, even if it means giving a less accurate answer.
"""

main_agent = create_agent(
    model=ChatOllama(model="qwen3.5:9b"),
    tools=[get_energy_usage_last_24h, get_model_energy_costs] + weather_tools,
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[tracker],
)

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Amsterdam?"}]}
)


print(response["messages"][-1].content)

print(f"Current estimated energy usage: {tracker.get_report()}")

