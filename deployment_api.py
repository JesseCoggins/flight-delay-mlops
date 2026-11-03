"""Root ASGI entry point for the flight-delay prediction service."""
from src.deployment_api import (
    airports, app, create_airport_encoding, model, predict_delays, predict_flight,
    root, time_to_seconds,
)
