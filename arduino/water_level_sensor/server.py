from flask import Flask, request, jsonify
import psycopg2

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)

cur = conn.cursor()

@app.route('/data', methods=['POST'])
def data():
    json_data = request.json
    print(request.json)

    water_level = json_data["water"]
    lat = json_data["lat"]
    lng = json_data["lng"]

    cur.execute(
        "INSERT INTO flood_data (water_level, lat, lng) VALUES (%s, %s, %s)",
        (water_level, lat, lng)
    )

    conn.commit()

    danger = water_level >= 1000

    return jsonify({
        "success": True,
        "danger": danger
    })

    # with open("data.txt", "a",encoding="utf-8") as f:
    #     f.write(str(json_data) + "\n")

    # return "ok"

app.run(host='0.0.0.0', port=5000)