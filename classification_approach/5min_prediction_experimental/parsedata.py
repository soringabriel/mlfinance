import json
import csv
import math
import requests

past_data_size = 480

data = []
links = [
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/314", # 2yUS
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/313", # 5yUS
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/312", # 10yUS
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/310", # RTY
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/29", # DJ30
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/27", # SPX
    "https://candle.etoro.com/candles/desc.json/FiveMinutes/1000/28", # Nasdaq
]
rows = []
for link in links:
    etoroData = requests.get(link)
    data = json.loads(etoroData.text)
    data = data['Candles'][0]['Candles']
    data.reverse()

    differences = []
    prev = None

    # Class 0 = unlikely - no difference
    # Class -1 = (0, -1%)
    # Class -2 = (-1%, -2%)
    # Class -3 = (-2%, -3%)
    # Class -4 = (-3%, -4%)
    # Class -5 = (-4%, -5%)
    # Class -6 = more than -5%
    # Same for the plus classes

    for v in data:
        if prev != None:
            dif = v['Close'] - prev
            percentage = dif * 100 / prev
            differences.append(percentage)
        prev = v['Close']

    train_data = differences[past_data_size:]
    index = 0
    size = past_data_size
    for v in train_data:
        row = differences[index:size]
        clasification = math.ceil(v) if v > 0 else math.floor(v)
        if v > 5:
            clasification = 6
        if v < -5:
            clasification = -6
        row.append(clasification)
        index += 1
        size += 1
        rows.append(row)
    
toPredict = [rows.pop()[:past_data_size]]
    
with open("data.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

with open("topredict.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(toPredict)