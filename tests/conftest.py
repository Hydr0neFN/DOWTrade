import pytest
from src.db.repo import Database


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _nim_disabled_by_default(monkeypatch):
    """No test may reach NVIDIA NIM by accident.

    The production box has a real key in ~/.nvidia-nim-key, so without this the
    ordinary LLM tests would send live requests (and spend the shared rate
    limit). Tests that exercise NIM install their own client with nim.set_client().
    """
    from src.llm import nim
    monkeypatch.delenv(nim.KEY_ENV, raising=False)
    nim.set_client(None)
    nim.GUARD.clear()
    yield
    nim.reset()
