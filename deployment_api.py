#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import pickle
import datetime

# Import the airport encodings file
f = open('airport_encodings.json')
 
# returns JSON object as a dictionary
airports = json.load(f)

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.  

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport
        
    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None

# TODO:  
# write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded 
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE



def time_to_seconds(time_str: str) -> int:
    """Convert time string in HH:MM format to seconds since midnight"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        # Add validation
        if not (0 <= hours <= 23) or not (0 <= minutes <= 59):
            raise ValueError(f"Invalid time format: {time_str}")
        return hours * 3600 + minutes * 60
    except (ValueError, IndexError):
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")

def predict_flight(dest_airport: str, departure_time: str, arrival_time: str, polynomial_order: int = 1):
    """Make prediction using the trained model"""
    # Create airport encoding for destination airport
    airport_encoding = create_airport_encoding(dest_airport, airports)
    if airport_encoding is None:
        raise ValueError(f"Destination airport code '{dest_airport}' not found")
    
    # Convert times to seconds since midnight
    dep_seconds = time_to_seconds(departure_time)
    arr_seconds = time_to_seconds(arrival_time)
    
    # Create feature array: 
    features = np.concatenate([[polynomial_order], airport_encoding, [dep_seconds, arr_seconds]])
    
    # Make prediction
    prediction = model.predict([features])[0]
    return prediction


# Load the trained model
with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



# TODO: 
#  write the API endpoints.  
# YOUR CODE GOES HERE

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint indicating API is functional"""
    return {"message": "API is functional"}

@app.get("/predict/delays")
async def predict_delays(dest_airport: str, departure_time: str, arrival_time: str):
    """Endpoint to predict departure delays"""
    try:
        prediction = predict_flight(
            #org_airport=org_airport,
            dest_airport=dest_airport,
            departure_time=departure_time,
            arrival_time=arrival_time,
            polynomial_order=1
        )
        return {
            "arrival_airport": dest_airport,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "average_departure_delay_minutes": float(np.asarray(prediction).reshape(-1)[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
