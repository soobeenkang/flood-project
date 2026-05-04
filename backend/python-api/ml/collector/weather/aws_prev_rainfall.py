import requests
import pandas as pd
from datetime import datetime, timedelta
from config import API_KEY

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"

STATIONS = [400,401,403,413,414]


def get_aws_rainfall(time, station):

    params = {
        "var": "RN",
        "tm": time,
        "stn": station,
        "authKey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)

    return response.text


def parse_response(text):

    lines = text.split("\n")

    for line in lines:

        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split()

        if len(parts) >= 4:
            # RN_HR1
            return float(parts[6])

    return None


def collect_history(start_date, end_date):

    current = start_date

    records = []

    while current <= end_date:

        time_str = current.strftime("%Y%m%d%H%M")

        for station in STATIONS:

            try:

                text = get_aws_rainfall(time_str, station)

                rain = parse_response(text)

                if rain is not None:

                    records.append({
                        "time": time_str,
                        "station": station,
                        "rain_1h": rain
                    })

            except:
                print("error:", station, time_str, e)

        current += timedelta(hours=1)

    df = pd.DataFrame(records)

    return df


if __name__ == "__main__":

    start = datetime(2020,1,1,0)
    end = datetime(2023,12,31,23)

    df = collect_history(start,end)

    df.to_csv("data/prev_rainfall/aws_prev_rainfall.csv", index=False)

    print("AWS prev rainfall 저장 완료")