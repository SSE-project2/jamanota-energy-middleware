from middleware import Datapoint

# Placeholders. Will be replaced by actual tracking data.
PAST_24H_ENERGY_J = 10
ENERGY_BUDGET_J = 10.5

def get_total_energy_usage(outputs: list[Datapoint]) -> float:
    return sum(dp.estimated_energy_joule for dp in outputs)