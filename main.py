from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from middleware import EnergyMiddleware
from reporting import get_total_energy_usage
import random
from datetime import datetime, timedelta

# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

@tool("get_weather_report", description="Retrieve detailed hourly weather report for a city")
def get_weather(city: str) -> dict:
    days = []
    stations = ["AMS1", "AMS2", "AMS3"]
    models = ["ECMWF", "GFS", "HIRLAM"]
    
    start_date = datetime.today()
    total_days = 7

    for day_index in range(total_days):
        day_date = start_date + timedelta(days=day_index)
        hours = []

        for h in range(0, 24):
            # Base values for this hour
            base_temp = 17 + random.uniform(-4, 7)
            base_rain_prob = random.uniform(0, 1)
            base_wind = random.uniform(5, 35)

            for station in stations:
                for model in models:
                    hours.append({
                        "hour": f"{h:02d}:00",
                        "station_id": station,
                        "forecast_model": model,
                        "temperature_c": round(base_temp + random.uniform(-1.0, 1.0), 1),
                        "feels_like_c": round(base_temp + random.uniform(-2.0, 2.0), 1),
                        "rain_probability": round(
                            min(max(base_rain_prob + random.uniform(-0.1, 0.1), 0), 1), 2
                        ),
                        "rain_mm": round(random.uniform(0, 8) * base_rain_prob, 1),
                        "wind_kmh": round(base_wind + random.uniform(-3, 3), 1),
                        "wind_gust_kmh": round(base_wind + random.uniform(5, 15), 1),
                        "humidity": random.randint(55, 98),
                        "visibility_km": round(random.uniform(2, 10), 1),
                        "cloud_cover_percent": random.randint(5, 100),
                        "conditions": random.choice([
                            "clear",
                            "cloudy",
                            "light rain",
                            "showers",
                            "overcast",
                            "mist"
                        ]),
                        "confidence": round(random.uniform(0.7, 0.98), 2)
                    })

        days.append({
            "date": day_date.strftime("%Y-%m-%d"),
            "hours": hours
        })

    return {
        "city": city,
        "metadata": {
            "generated_by": "SyntheticWeatherSim v3.0",
            "forecast_models_used": models,
            "data_points": sum(len(day['hours']) for day in days),
            "generation_time": "simulated"
        },
        "units": {
            "temperature": "C",
            "rain": "mm",
            "wind": "km/h",
            "visibility": "km"
        },
        "daily_summary": {
            "sunrise": "06:48",
            "sunset": "18:31",
            "uv_index": random.randint(1, 5),
            "pressure_hpa": random.randint(1005, 1025)
        },
        "forecast_days": days
    }

SUBAGENT_SYSTEM_PROMPT = """
You are a meteorological analysis assistant.

You receive detailed structured weather reports produced by multiple weather
stations and forecast models. For each hour there may be multiple observations
from different stations and models.

Each observation may contain:
- temperature and feels_like temperature
- rain probability and expected rainfall
- wind speed and wind gusts
- humidity and visibility
- cloud cover and general conditions
- the station ID and forecast model that produced the observation
- a confidence score

The data may contain minor variations between stations and models. Your task
is to interpret the overall weather situation rather than focusing on a single
data point.

When answering a user question:

1. Identify the relevant time range mentioned in the question
   (for example: morning, afternoon, evening, tonight, or a specific hour).

2. For each relevant hour, consider multiple observations from different
   stations and forecast models and determine the overall trend or consensus.

3. Pay particular attention to:
   - rain probability and rainfall amount
   - wind speed and gusts
   - visibility
   - temperature and feels-like temperature
   - any weather alerts

4. Ignore irrelevant metadata such as station IDs unless it helps explain
   uncertainty in the forecast.

5. Summarize the weather clearly and concisely in natural language.

Focus on giving a useful interpretation of the data rather than repeating
the raw values.
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
    "large": create_weather_subagent(MODEL_TIERS["large"]),
    "medium": create_weather_subagent(MODEL_TIERS["medium"]),
    "small": create_weather_subagent(MODEL_TIERS["small"]),
}

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


MAIN_SYSTEM_PROMPT = f"""
You are an assistant that answers user questions while managing energy consumption.

You do NOT directly analyze weather data yourself. Instead, you must decide
which specialized weather analysis agent to call.

Available weather analysis agents:
- weather_small  : lowest energy cost, suitable for simple questions
- weather_medium : moderate energy cost, suitable for moderately complex analysis
- weather_large  : highest energy cost, suitable for complex reasoning over large weather reports

Decision strategy:

1. Analyze the user's question and estimate its complexity. Consider aspects such as:
   - Does the question ask for a simple summary of the weather?
   - Does it require analyzing multiple hours of forecast data?
   - Does it require comparing wind, rain, and temperature to make a recommendation?
   - Does it require identifying trends or optimal times during the day?

2. Estimate how much reasoning and analysis will be required.

3. Select the cheapest weather agent that can reliably answer the question given its estimated complexity.

General complexity guidelines:

Low complexity:
- Simple weather summaries
- Questions about current temperature or conditions
→ Use weather_small

Medium complexity:
- Questions about weather conditions during a small time range
- Requires scanning several hours of forecast
→ Use weather_medium

High complexity:
- Recommendation questions
- Requires comparing multiple weather variables across multiple hours
→ Use weather_large

Today is {datetime.today()}. Always try to minimize energy usage while still providing a useful answer.
"""

main_agent = create_agent(
    model=ChatOllama(model="qwen3.5:9b"),
    tools=weather_tools,
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[tracker],
)

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like today between 2 PM and 4 PM?"}]}
)


print(response["messages"][-1].content)

print(f"Current estimated energy usage: {tracker.get_report()}")

