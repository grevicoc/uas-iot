# from fastapi import FastAPI, Body
# import uvicorn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import paho.mqtt.client as mqtt
from datetime import datetime
import re
import pickle
import pandas as pd
import json

model = pickle.load(open('./model.pkl', 'rb'))

RAW_DATA_TOPIC = "raw_data"
RESULT_DATA_TOPIC = "result_topic"

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(RAW_DATA_TOPIC)


time_scaled = None
light = None
hum = None
temp = None
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global time_scaled
    global light
    global hum
    global temp

    payload_str = msg.payload.decode("utf-8")
    print("Received message:", payload_str)

    if "Light" in payload_str:
        splitted_str = payload_str.split(" ")
        # print(splitted_str)

        date_str = splitted_str[0]
        light_str = splitted_str[len(splitted_str)-1]

        # print(date_str)
        # print(light_str)

        date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M')
        time_scaled = (float(date.hour) + (float(date.minute) / 60.0)) / 23.9
        light = int(light_str)

        # print(time_scaled)
        # print(light)

        produce_result(client)

    if "Temp" in payload_str:
        splitted_str = payload_str.split(" ")
        print(splitted_str)

        temp_str = splitted_str[3]
        hum_str = splitted_str[len(splitted_str)-1]

        pattern = r'\d+\.\d+|\d+'
        matches = re.findall(pattern, temp_str)

        hum = int(float(hum_str) * 1000)
        temp = int(float(matches[0]) * 1000)

        # print(hum)
        # print(temp)

        produce_result(client)

def produce_result(client):
    global time_scaled
    global light
    global hum
    global temp

    if time_scaled != None and light != None and hum != None and temp != None:
        input_df = pd.DataFrame({'Temp': [temp], 'Hum': [hum], 'Light': [light]})
        input_df = input_df.reindex(columns=['Hum', 'Light', 'Temp'])

        prediction = model.predict(input_df)
        print("Prediction:", prediction)

        class_probabilities = model.predict_proba(input_df)
        confidence = np.max(class_probabilities)
        print("Confidence:", confidence)

        data = {
            "prediction_value": str(prediction[0]),
            "confidence_value": str(confidence )
        }

        json_payload = json.dumps(data)

        client.publish(RESULT_DATA_TOPIC, json_payload) # Sends either "0" or "1"

        time_scaled = None
        light = None
        hum = None
        temp = None

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("0.tcp.ap.ngrok.io", 12993, 60)
client.loop_forever()

# app = FastAPI()

# @app.post("/predict")
# async def predict(
#     time: str = Body(...),
#     temp: int = Body(...),
#     hum: int = Body(...),
#     light: int = Body(...)
# ):
    
    # raw_input = {"time": time, "temp": temp, "hum": hum, "light": light}
    # print(raw_input)

    # scaler = MinMaxScaler(**scaler_params)
    # scaled_input = scaler.transform([[raw_input['temp'], raw_input['hum'], raw_input['light']]])
    # print(scaled_input)

    # # Preprocess input data as necessary
    # data = np.array([[time, temp, hum, light]])
    
    # # Pass data to the model for prediction
    # prediction = model.predict(data)
    
    # # Format prediction results as a response
    # response = {
    #     "will_rain": prediction[0]
    # }
    
#     return response

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)