"""Testy dla Solution."""
from uav_moea.io.data_loader import load_test_case
from uav_moea.model.individual import Solution


TC_PATH = "data/TC-PGI/tcB.json"


class TestSolution:
    def test_create_solution(self):
        sol = Solution(paths=[[0, 1, 2], [3, 4, 5]], num_uavs=2)

        assert sol.num_uavs == 2
        assert len(sol.paths) == 2
        assert sol.f1 is None
        assert sol.f2 is None

    def test_validate_correct(self):
        tc = load_test_case(TC_PATH)
        regions = list(range(tc.node_count))
        mid = tc.node_count // 2

        sol = Solution(paths=[regions[:mid], regions[mid:]], num_uavs=2)

        assert sol.validate(tc) is True

    def test_validate_missing_region(self):
        tc = load_test_case(TC_PATH)
        regions = list(range(tc.node_count))
        mid = tc.node_count // 2

        # Usuń jeden region
        sol = Solution(paths=[regions[:mid], regions[mid + 1:]], num_uavs=2)

        assert sol.validate(tc) is False

    def test_validate_wrong_num_uavs(self):
        tc = load_test_case(TC_PATH)
        regions = list(range(tc.node_count))

        sol = Solution(paths=[regions], num_uavs=2)  # 1 ścieżka, ale num_uavs=2

        assert sol.validate(tc) is False

    def test_copy_is_independent(self):
        sol = Solution(paths=[[0, 1, 2], [3, 4, 5]], num_uavs=2)
        sol.f1 = 10.0
        sol.f2 = 0.5

        clone = sol.copy()
        clone.paths[0][0] = 99
        clone.f1 = 999.0

        assert sol.paths[0][0] == 0  # oryginał niezmieniony
        assert sol.f1 == 10.0

    def test_repr(self):
        sol = Solution(paths=[[0, 1], [2, 3, 4]], num_uavs=2)

        text = repr(sol)

        assert "UAVs=2" in text
        assert "[2, 3]" in text