import os
import tempfile

import pandas as pd
import pytest

from src.app_csv import create_app


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
        csv_path = tmp_csv.name

    app = create_app(config={"CSV_PATH": csv_path})
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client, csv_path

    os.remove(csv_path)

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
