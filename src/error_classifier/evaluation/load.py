import sys
from collections.abc import Callable
from typing import Final

from error_classifier.evaluation.metrics import LOAD_LIMIT

LOAD_WAIT_S: Final = 600
LOAD_STEP_S: Final = 15

type LoadAvg = Callable[[], tuple[float, float, float]]


def wait_for_load(sleep: Callable[[float], None], loadavg: LoadAvg) -> tuple[float, float, float]:
    current = loadavg()
    waited = 0
    while current[0] >= LOAD_LIMIT and waited < LOAD_WAIT_S:
        print("waiting for load ...", file=sys.stderr)
        sleep(LOAD_STEP_S)
        waited += LOAD_STEP_S
        current = loadavg()
    return current
