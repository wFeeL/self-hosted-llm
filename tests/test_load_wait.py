import pytest

from error_classifier.evaluation.load import LOAD_STEP_S, LOAD_WAIT_S, wait_for_load


class Load:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.pauses: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)

    def loadavg(self) -> tuple[float, float, float]:
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        return (value, 0.0, 0.0)


def test_quiet_machine_does_not_wait() -> None:
    load = Load([0.5])
    assert wait_for_load(load.sleep, load.loadavg) == (0.5, 0.0, 0.0)
    assert load.pauses == []


def test_waits_until_load_drops(capsys: pytest.CaptureFixture[str]) -> None:
    load = Load([3.0, 2.0, 1.0])
    assert wait_for_load(load.sleep, load.loadavg)[0] == 1.0
    assert load.pauses == [LOAD_STEP_S, LOAD_STEP_S]
    assert capsys.readouterr().err.count("waiting for load") == 2


def test_gives_up_after_the_limit() -> None:
    load = Load([3.0])
    assert wait_for_load(load.sleep, load.loadavg)[0] == 3.0
    assert sum(load.pauses) == LOAD_WAIT_S
