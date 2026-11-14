"""Inner harnesses: four ways for a model to spell the same tool calls."""

from mhrl.harnesses.base import Harness, ParseError
from mhrl.harnesses.json_calls import JsonCallsHarness
from mhrl.harnesses.plaintext import PlainTextHarness
from mhrl.harnesses.pycall import PyCallHarness
from mhrl.harnesses.react_xml import ReactXmlHarness

TRAIN_HARNESSES: tuple[str, ...] = ("json", "react_xml", "pycall")
HELD_OUT_HARNESSES: tuple[str, ...] = ("plaintext",)

_REGISTRY: dict[str, type[Harness]] = {
    h.name: h for h in (JsonCallsHarness, ReactXmlHarness, PyCallHarness, PlainTextHarness)
}


def get_harness(name: str) -> Harness:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown harness {name!r}; choose from {sorted(_REGISTRY)}") from None


def all_harnesses() -> list[Harness]:
    return [get_harness(n) for n in (*TRAIN_HARNESSES, *HELD_OUT_HARNESSES)]


__all__ = [
    "HELD_OUT_HARNESSES",
    "TRAIN_HARNESSES",
    "Harness",
    "ParseError",
    "all_harnesses",
    "get_harness",
]
