import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

### 변화 가능 12-24?
SEQ_LEN = 12


def make_lstm_data():

    input_dir = "data/final_by_year_merged"
    output_dir = "data/lstm_data"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    #   --- train/test 연도기준 나누기
    train_files = [f for f in files if int(f.split("_")[1].split(".")[0]) <= 2022]
    test_files = [f for f in files if int(f.split("_")[1].split(".")[0]) >= 2023]
    
    print("train: ", train_files)
    print("test: ", test_files)
    
    #   --- scaler
    scaler = StandardScaler()

    print("\n[1] scaler 학습 중")

    for file in train_files:
        print(f"\n처리중: {file}")

        df = pd.read_parquet(f"{input_dir}/{file}")
        df = df.sort_values(["grid_id", "time"])

        # 새로운 피쳐 만들기
        df["rain_diff"] = df.groupby("grid_id")["rain_1h"].diff().fillna(0)
        df["prev_flood"] = df.groupby("grid_id")["flood"].shift(1).fillna(0)
        df["month"] = df["time"].dt.month

        feature_cols = [
            "rain_1h", "rain_3h", "rain_6h",
            "rain_intensity", "rain_max_3h",
            "rain_diff",
            "mean_elevation", "is_river",
            "month",
            "prev_flood"
        ]

        scaler.partial_fit(df[feature_cols])

    print("scaler 완료")


    # sequence 생성 함수
    def process_files(file_list, prefix):

        X_list, y_list = [], []
        chunk_idx = 0

        for file in file_list:
            print(f"\n처리중: {file}")

            df = pd.read_parquet(f"{input_dir}/{file}")
            df = df.sort_values(["grid_id", "time"])

            # 새로운 피쳐 만들기
            df["rain_diff"] = df.groupby("grid_id")["rain_1h"].diff().fillna(0)
            df["prev_flood"] = df.groupby("grid_id")["flood"].shift(1).fillna(0)
            df["hour"] = df["time"].dt.hour
            df["month"] = df["time"].dt.month

            feature_cols = [
                "rain_1h", "rain_3h", "rain_6h",
                "rain_intensity", "rain_max_3h",
                "rain_diff",
                "mean_elevation", "is_river",
                "hour", "month",
                "prev_flood"
            ]

            # feature scaling
            df[feature_cols] = scaler.transform(df[feature_cols])

            for grid_id, group in df.groupby("grid_id"):
                group = group.sort_values("time")

                values = group[feature_cols].values
                labels = group["flood"].values

                # --- sequence 생성 핵심
                for i in range(len(group) - SEQ_LEN):
                    X_list.append(values[i:i+SEQ_LEN])
                    y_list.append(labels[i+SEQ_LEN])

            print(f"현재 샘플 수: {len(X_list)}")

            # 너무 커지면 저장
            if len(X_list) > 300000:
                save_chunk(X_list, y_list, output_dir, prefix, chunk_idx)
                chunk_idx += 1
                X_list, y_list = [], []

        # 남은 부분 저장
        if len(X_list) > 0:
            save_chunk(X_list, y_list, output_dir, prefix, chunk_idx)

    process_files(train_files, "train")
    process_files(test_files, "test")
        
    print("LSTM 데이터 생성 완료")


def save_chunk(X_list, y_list, output_dir, prefix, idx):

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)

    np.save(f"{output_dir}/{prefix}_X_{idx}.npy", X)
    np.save(f"{output_dir}/{prefix}_y_{idx}.npy", y)

    print(f"저장 완료: {prefix} chunk {idx}, shape={X.shape}")


if __name__ == "__main__":
    make_lstm_data()