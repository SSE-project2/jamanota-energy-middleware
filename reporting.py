from middleware import Datapoint

def get_total_energy_usage(outputs: list[Datapoint]) -> float:
    return sum(dp.estimated_energy_joule for dp in outputs)

def get_total_co2(outputs: list[Datapoint]) -> float:
    return sum(dp.estimated_co2e_gram for dp in outputs)