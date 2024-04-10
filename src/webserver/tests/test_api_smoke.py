from flask.testing import FlaskClient
from pathlib import Path


# get the resources folder in the tests folder
resources = Path(__file__).parent / "resources"


def test_swagger(client: FlaskClient):
    response = client.get("/")

    assert response.status_code == 200


def test_predict_should_return_400_when_type_it_empty(client: FlaskClient):
    response = client.post("/predict")

    assert response.status_code == 400


def test_predict_should_return_200_when_type_is_invalid(client: FlaskClient):
    response = client.post("/predict?type=invalid", data={
        "image": (resources / "walls" / "F1_original.png").open("rb"),
    })

    assert response.status_code == 400


def test_predict_walls_should_return_200_when_type_and_image_is_provided(client: FlaskClient):
    response = client.post("/predict?type=walls", data={
        "image": (resources / "walls" / "F1_original.png").open("rb"),
    })

    assert response.status_code == 200


def test_predict_tables_should_return_200_when_type_and_image_is_provided(client: FlaskClient):
    response = client.post("/predict?type=tables", data={
        "image": (resources / "tables" / "image_1.jpg").open("rb"),
    })

    assert response.status_code == 200
