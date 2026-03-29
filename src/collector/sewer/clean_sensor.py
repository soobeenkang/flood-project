from pathlib import Path
import pandas as pd

#location.py 돌리면 csv에 locaition column이 같이 저장 돼서 그거 없애는 코드
BASE_DIR = Path(__file__).resolve().parents[3]

path = BASE_DIR / "data" / "sensor" / "sensor_locations.csv"

df = pd.read_csv(path)

df = df.drop(columns=["location"])

df.to_csv(path, index=False, encoding="utf-8-sig")

print("location 컬럼 삭제 완료")