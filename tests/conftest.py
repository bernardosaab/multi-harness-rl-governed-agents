from __future__ import annotations

import pytest

from mhrl.domain.tasks import Task, generate_tasks


@pytest.fixture(scope="session")
def tasks() -> list[Task]:
    return generate_tasks()


@pytest.fixture(scope="session")
def by_id(tasks: list[Task]) -> dict[str, Task]:
    return {t.id: t for t in tasks}
