from pathlib import Path

import pytest

from tests.fixtures.generate_geometry_fixtures import create_fixtures


@pytest.fixture(scope="session", autouse=True)
def geometry_fixtures() -> None:
    create_fixtures()


@pytest.fixture(scope="session")
def fixture_directory() -> Path:
    return Path(__file__).parent / "fixtures"

