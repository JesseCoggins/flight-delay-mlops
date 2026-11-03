#!/usr/bin/env python
# coding: utf-8

from fastapi.testclient import TestClient

from deployment_api import app
from src.deployment_api import app as packaged_app

# Create test client
client = TestClient(app)


def test_root():
    """Testing the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is functional"}

def test_valid_prediction():
    """Testing the /predict/delays endpoint with valid parameters"""
    response = client.get("/predict/delays", params={
        "dest_airport": "LAX",
        "departure_time": "14:30",
        "arrival_time": "17:45"
    })
    assert response.status_code == 200
    data = response.json()
    assert "average_departure_delay_minutes" in data

def test_invalid_airport():
    """Testing the /predict/delays endpoint with an invalid airport"""
    response = client.get("/predict/delays", params={
        "dest_airport": "INVALID",
        "departure_time": "14:30",
        "arrival_time": "17:45"
    })
    assert response.status_code == 400

def test_invalid_time_format():
    """Testing the /predict/delays endpoint with invalid time format"""
    response = client.get("/predict/delays", params={
        "dest_airport": "LAX",
        "departure_time": "25:30",  # Invalid hour
        "arrival_time": "17:45"
    })
    assert response.status_code == 400

def test_missing_parameters():
    """Testing the /predict/delays endpoint with missing parameters"""
    response = client.get("/predict/delays", params={
        "dest_airport": "LAX"
        # Missing departure_time and arrival_time
    })
    assert response.status_code == 422
    


def test_entry_points_and_airport_aliases_agree():
    assert app is packaged_app
    params = {"departure_time": "14:30", "arrival_time": "17:45"}
    codes = ["LAX", "12892"]
    responses = [client.get("/predict/delays", params={**params, "dest_airport": code}) for code in codes]
    assert [response.status_code for response in responses] == [200, 200]
    delays = [response.json()["average_departure_delay_minutes"] for response in responses]
    assert delays[0] == delays[1]
