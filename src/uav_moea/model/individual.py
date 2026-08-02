"""
Reprezentacja rozwiązania — zestaw ścieżek UAV.
"""

from __future__ import annotations

from uav_moea.io.data_loader import TestCase

class Solution:
    """Rozwiązanie - zestaw ścieżek dla wielu dronów"""

    def __init__(self, paths:list[list[int]], num_uavs: int):
        self.paths = paths
        self.num_uavs = num_uavs
        self.f1: float | None = None
        self.f2: float | None = None

    def validate(self, test_case: TestCase) -> bool:
        """Sprawdzamy czy rozwiazanie jest poprawne (kazdy region odwiedzany dokladnie 1 raz)"""
        if len(self.paths) != self.num_uavs:
            return False
        
        all_regions = []
        for path in self.paths:
            all_regions.extend(path)

        if len(all_regions) != test_case.node_count:
            return False
        
        if set(all_regions) != set(range(test_case.node_count)):
            return False
        
        return True
    
    def copy(self) -> Solution:
        """Głeboka kopia rozwiazania"""
        clone = Solution(
            paths=[p.copy() for p in self.paths],
            num_uavs = self.num_uavs,
        )
        clone.f1 = self.f1
        clone.f2 = self.f2

        return clone
    
    def __repr__(self) -> str:
        regions_per_uav = [len(path) for path in self.paths]
        return f"Solution(UAVs={self.num_uavs}, regions={regions_per_uav}, f1 = {self.f1}, f2 = {self.f2})"