from __future__ import annotations

from datetime import datetime

import pytest

from cooking_assistant_ai.core.clock import Clock
from cooking_assistant_ai.core.tools import ToolContext
from cooking_assistant_ai.model.types import Session
from cooking_assistant_ai.storage.db import Store

T0 = datetime(2026, 9, 3, 18, 30)


@pytest.fixture
def clock() -> Clock:
    return Clock(T0)


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def session(store: Store, clock: Clock) -> Session:
    s = Session(id="test", started_at=clock.now())
    for rid in ("r001", "r002", "r003"):
        s.add_recipe(store.get_recipe(rid))
    return s


@pytest.fixture
def ctx(session: Session, clock: Clock, store: Store) -> ToolContext:
    # A fully equipped test kitchen, so tests are about the thing they are testing rather
    # than about what the cook happens to own. Tests for the ownership constraint itself
    # call store.set_appliances explicitly.
    from cooking_assistant_ai.core.appliances import CATALOGUE

    store.set_appliances(list(CATALOGUE))
    return ToolContext(session, clock, store)
