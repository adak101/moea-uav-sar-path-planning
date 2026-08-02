"""
Ładowanie danych tekstowych z plików JSON

"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TestCase:
    """Przypadek testowy - mapa regionów z czasami przelotów i ofiarami"""

    name: str
    node_count: int
    node_population: int
    time_process_node: list[float]
    time_process_edge: list[list[float]]
    time_reach_node: list[float]

    def __post_init__(self):
        assert len(self.node_population) == self.node_count
        assert len(self.time_process_node) == self.node_count
        assert len(self.time_reach_node) == self.node_count
        assert len(self.time_process_edge) == self.node_count
        assert all(len(row) == self.node_count for row in self.time_process_edge)


    @property
    def non_empty_regions(self) -> list[float]:
        """Regiony z ofiarami (population > 0)"""
        return [i for i, pop in enumerate(self.node_population) if pop > 0]
    
    @property
    def total_population(self) -> int:
        """Łączna liczba poszkodowanych"""
        return sum(self.node_population)
    
    def get_flight_time(self, from_region: int, to_region: int) -> float:
        return self.time_process_edge[from_region][to_region]
    
    def get_scan_time(self, region: int)-> float:
        return self.time_process_node[region]
    
    def get_reach_time(self, region: int) -> float:
        return self.time_reach_node[region]
    
    def get_population(self, region: int) -> int:
        return self.node_population[region]
    

def load_test_case(filepath: str | Path) -> TestCase:
    """Załaduj przypadek testowy z pliku JSON."""
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return TestCase(
        name=filepath.stem,
        node_count=data["node-count"],
        node_population=data["node-population"],
        time_process_node=data["time-process-node"],
        time_process_edge=data["time-process_edge"],
        time_reach_node=data["time-reach-node"],
    )

def load_all_test_cases(data_dir: str | Path = "data") -> dict[str, TestCase]:
    """Załaduj wszystkie plik json z katalogu"""
    data_dir = Path(data_dir)

    result = {}
    for json_file in data_dir.rglob("*.json"):
        tc = load_test_case(json_file)
        result[tc.name] = tc

    return result


