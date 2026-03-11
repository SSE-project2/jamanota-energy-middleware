import uuid
from collections import defaultdict

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from context_var import prompt_id_var
from middleware import EnergyMiddleware, CustomState
from reporting import get_total_energy_usage, get_total_co2


# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

@tool("get_weather", description="Get the weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is rainy and 21.7 degrees Celsius"

SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant. You are an expert at researching the weather. Respond in a whimsical tone.
You have the following tools:
- get_weather: this tool takes a city as input and returns the weather in that city.
"""

tracker = EnergyMiddleware()

subagent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[get_weather],
    system_prompt=SUBAGENT_SYSTEM_PROMPT,
    middleware=[tracker],
    state_schema=CustomState
)

@tool("weather", description="Research the weather and return findings")
def call_weather_agent(query: str, prompt_id: str) -> str:
    print(f'prompt_id: {prompt_id}')
    result = subagent.invoke({"messages": [{"role": "user", "content": query}], 'prompt_id': prompt_id})
    return result["messages"][-1].content

MAIN_SYSTEM_PROMPT = """
You are a helpful assistant. Respond in a serious tone.

You have access to the following tools:
- call_weather_agent: this calls another agent that will research the weather in a city when asked. Make sure to specify the name of the city.
"""

main_agent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[call_weather_agent],
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[tracker],
    state_schema=CustomState
)

# prompt_id_var.set(str(uuid.uuid4()))

uuid_val = uuid.uuid4()
print(uuid_val)
str_uuid = str(uuid_val)
print(str_uuid)

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Amsterdam?"}],
     'prompt_id': str_uuid}
)

print(response["messages"][-1].content)

report = tracker.get_report()

print(report)

grouped = defaultdict(list)
for dp in report:
    grouped[dp.prompt_id].append(dp)

for prompt_id, points in grouped.items():
    print(f"\nPrompt [{prompt_id}]: {len(points)} calls")
    for dp in points:
        total_energy = dp.estimated_energy_joule
        total_co2e = dp.estimated_co2e_gram
        print(f"  [{dp.model_name}] {dp.message}  {total_energy} J | {total_co2e} gCO2e") # If we have multiple models for the subprompts that will change here

