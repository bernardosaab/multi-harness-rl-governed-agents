"""writegate-rl: multi-harness RL for agents that write to a system of record.

The model proposes; the harness decides. Inner harnesses (prompt and tool-call
formats) vary, while one outer harness (the write gate) binds every change.
"""

__version__ = "0.1.0"
