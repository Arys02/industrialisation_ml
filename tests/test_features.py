import os
import tempfile

import pandas as pd
import pytest

from src.features import features


@pytest.fixture
def sales_data():
    return pd.DataFrame({
        'sales': []
    })


def test_hello(client):
    client, csv_path = client
    response = client.get('/hello')
    assert response.status_code == 200
    assert response.data == b'hello'

def test_post_data(client):
    client, csv_path = client

    response = client.post('/post_data', json=[{"year_week": 202002, "vegetable": "tomato", "sales": 100}])
    assert response.status_code == 200
    assert response.json == {"status": "success"}

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "vegetable" in df.columns
    assert "tomato" in df["vegetable"].values
