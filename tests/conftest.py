import pytest
from src.db.repo import Database


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()
