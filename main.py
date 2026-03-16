from agents import main_agent, tracker

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Amsterdam?"}]}
)

print(response["messages"][-1].content)
print(f"Current estimated energy usage: {tracker.get_report()}")