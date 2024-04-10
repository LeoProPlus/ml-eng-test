
import pytest
from webserver.app import app


@pytest.fixture()
def client():
    return app.test_client()
