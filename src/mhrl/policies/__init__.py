"""Policies turn conversations into text. The runtime never knows which one it has."""

from mhrl.policies.base import Generation, Policy
from mhrl.policies.scripted import ScriptedPolicy

__all__ = ["Generation", "Policy", "ScriptedPolicy"]
