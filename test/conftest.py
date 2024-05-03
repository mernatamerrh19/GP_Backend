from functools import partial
import pytest
from rest_framework.test import APIClient


@pytest.fixture(autouse=True, scope="session")
def client():
    client = APIClient()
    client.post = partial(client.post, format="json")
    return client
