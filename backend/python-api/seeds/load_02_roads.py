import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

df = pd.read_csv(
    "data/서울시 자치구별 도보 네트워크 공간정보.csv",
    encoding="cp949"
)

required_cols = ["링크 ID", "시작노드 ID", "종료노드 ID", "링크 WKT", "링크 길이"]

before_count = len(df)
df = df.dropna(subset=required_cols)
after_count = len(df)

print(f"전체 행 수: {before_count}")
print(f"결측치 제거 후 행 수: {after_count}")
print(f"제거된 행 수: {before_count - after_count}")

df["링크 ID"] = df["링크 ID"].astype(int)
df["시작노드 ID"] = df["시작노드 ID"].astype(int)
df["종료노드 ID"] = df["종료노드 ID"].astype(int)
df["링크 길이"] = df["링크 길이"].astype(float)

with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO road_edge (
                edge_id, from_node, to_node, geom, distance_m, road_type
            )
            VALUES (
                :edge_id, :from_node, :to_node,
                ST_GeomFromText(:wkt, 4326),
                :distance_m, :road_type
            )
            ON CONFLICT (edge_id) DO NOTHING;
        """), {
            "edge_id": row["링크 ID"],
            "from_node": row["시작노드 ID"],
            "to_node": row["종료노드 ID"],
            "wkt": row["링크 WKT"],
            "distance_m": row["링크 길이"],
            "road_type": str(row["링크 유형 코드"]) if pd.notna(row["링크 유형 코드"]) else None
        })

print("road_edge 적재 완료")