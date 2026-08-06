import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n[1] Testing GET /health ...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        print("[✓] Health check passed!")
    except Exception as e:
        print(f"[✗] Health check failed: {e}")


def test_regression():
    print("\n[2] Testing POST /predict/regression ...")
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    try:
        res = requests.post(f"{BASE_URL}/predict/regression", json=payload)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        print("[✓] Regression endpoint passed!")
    except Exception as e:
        print(f"[✗] Regression endpoint failed: {e}")


def test_classification():
    print("\n[3] Testing POST /predict/classification ...")
    payload = {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184
    }
    try:
        res = requests.post(f"{BASE_URL}/predict/classification", json=payload)
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        print("[✓] Classification endpoint passed!")
    except Exception as e:
        print(f"[✗] Classification endpoint failed: {e}")


if __name__ == "__main__":
    test_health()
    test_regression()
    test_classification()
    print("\n✨ All API tests completed!")
