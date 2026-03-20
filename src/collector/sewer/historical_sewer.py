import zipfile
from pathlib import Path
import pandas as pd

# 프로젝트 루트 기준 경로
BASE_DIR = Path(__file__).resolve().parents[3]

# 헤더 통합 -> STANDARD_COLUMNS가 기준
STANDARD_COLUMNS = [
    "unq_no",
    "se_cd",
    "se_nm",
    "msrmt_ymd",
    "msrmt_watl",
    "sgn_stts",
    "pstn_info",
    "source_file",
]

COLUMN_MAP = {
    "UNQ_NO": "unq_no",
    "unq_no": "unq_no",
    "고유번호": "unq_no",

    "SE_CD": "se_cd",
    "se_cd": "se_cd",
    "구분코드": "se_cd",

    "SE_NM": "se_nm",
    "se_nm": "se_nm",
    "구분명": "se_nm",

    "MSRMT_YMD": "msrmt_ymd",
    "msrmt_ymd": "msrmt_ymd",
    "측정일자": "msrmt_ymd",

    "MSRMT_WATL": "msrmt_watl",
    "msrmt_watl": "msrmt_watl",
    "측정수위": "msrmt_watl",

    "SGN_STTS": "sgn_stts",
    "sgn_stts": "sgn_stts",
    "통신상태": "sgn_stts",

    "PSTN_INFO": "pstn_info",
    "pstn_info": "pstn_info",
    "위치정보": "pstn_info",
}


def standardize_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["source_file"] = source_file
    return df[STANDARD_COLUMNS]


# 헤더 없는 옛날 csv 파일 읽기 (2012~2018)
def read_old_csv_file(file_obj, source_file: str) -> pd.DataFrame:
    old_cols = [
        "unq_no",
        "legacy_se_cd",
        "se_nm",
        "legacy_time",
        "msrmt_ymd",
        "msrmt_watl",
        "sgn_stts",
    ]

    try:
        df = pd.read_csv(
            file_obj,
            header=None,
            names=old_cols,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(
            file_obj,
            header=None,
            names=old_cols,
            encoding="cp949",
            engine="python",
            on_bad_lines="skip",
        )

    # 표준 컬럼 맞추기
    df["se_cd"] = df["legacy_se_cd"]
    df["pstn_info"] = None

    return standardize_columns(df, source_file)


# 헤더 있는 최근 csv 파일 읽기 (2019~)
def read_csv_file(file_obj, source_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            file_obj,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(
            file_obj,
            encoding="cp949",
            engine="python",
            on_bad_lines="skip",
        )

    return standardize_columns(df, source_file)


# 강남구 데이터만 필터링
def filter_gangnam(df: pd.DataFrame) -> pd.DataFrame:
    df["se_cd"] = df["se_cd"].astype(str).str.strip()
    df["se_nm"] = df["se_nm"].astype(str).str.strip()
    return df[(df["se_cd"] == "23") | (df["se_nm"].str.contains("강남", na=False))]


# 데이터 자료형 변경
def clean_types(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df["msrmt_watl"] = pd.to_numeric(df["msrmt_watl"], errors="coerce")

    # 구형 데이터(2012~2018)는 예: 01-MAY-12 00:00:00
    if year <= 2018:
        df["msrmt_ymd"] = pd.to_datetime(
            df["msrmt_ymd"].astype(str).str.strip(),
            format="%d-%b-%y %H:%M:%S",
            errors="coerce",
        )
    else:
        df["msrmt_ymd"] = pd.to_datetime(
            df["msrmt_ymd"].astype(str).str.strip(),
            errors="coerce",
        )

    return df


# zip 안에 있는 csv 파일들을 읽는 함수
def load_zip_file(zip_path: str) -> pd.DataFrame:
    dfs = []

    # 폴더명(예: 2012) 기준으로 old/new 구분
    year = int(Path(zip_path).parent.name)

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            lower_name = name.lower()

            if lower_name.endswith(".csv") and not name.endswith("/"):
                print(f"읽는 중: {zip_path} -> {name}")

                with z.open(name) as f:
                    try:
                        if year <= 2018:
                            df = read_old_csv_file(
                                f,
                                source_file=f"{Path(zip_path).name}:{name}",
                            )
                        else:
                            df = read_csv_file(
                                f,
                                source_file=f"{Path(zip_path).name}:{name}",
                            )

                        print("원본 행 수:", len(df))

                        print(df[["unq_no", "se_cd", "se_nm", "msrmt_ymd", "msrmt_watl"]].head())

                        df = filter_gangnam(df)
                        print("강남 필터 후 행 수:", len(df))

                        df = clean_types(df, year)
                        dfs.append(df)

                    except pd.errors.EmptyDataError:
                        print(f"빈 파일 건너뜀: {name}")
                        continue
                    except Exception as e:
                        print(f"파일 읽기 실패: {name} | 에러: {e}")
                        continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame(columns=STANDARD_COLUMNS)


# 센서 데이터에 grid_id를 붙이고, 모델 입력용 형태로 바꾸는 함수
def attach_grid_and_transform(df: pd.DataFrame, grid_map_path: str) -> pd.DataFrame:
    grid_map_df = pd.read_csv(grid_map_path)

    # sensor id 타입/공백 정리
    df["unq_no"] = df["unq_no"].astype(str).str.strip()
    grid_map_df["sensor_id"] = grid_map_df["sensor_id"].astype(str).str.strip()

    # grid 붙이기
    merged = df.merge(
        grid_map_df[["sensor_id", "grid_id"]],
        left_on="unq_no",
        right_on="sensor_id",
        how="left",
    )

    # 필요한 컬럼만 추출
    result = merged[["msrmt_ymd", "grid_id", "msrmt_watl"]].copy()
    result = result.rename(columns={
        "msrmt_ymd": "time",
        "msrmt_watl": "water_level",
    })

    # grid 없는 행 제거
    result = result.dropna(subset=["grid_id"])

    # 타입 정리
    result["grid_id"] = result["grid_id"].astype("Int64")
    result["water_level"] = pd.to_numeric(result["water_level"], errors="coerce")

    # 이상값 제거
    result = result.dropna(subset=["water_level"])
    result = result[result["water_level"] >= 0]

    # 시간 단위로 정규화
    result["time"] = pd.to_datetime(result["time"], errors="coerce").dt.floor("h")

    # time NaT 제거
    result = result.dropna(subset=["time"])

    # 같은 시간 + 같은 grid면 최대 수위만 사용
    result = (
        result.groupby(["time", "grid_id"], as_index=False)["water_level"]
        .max()
    )

    # 정렬
    result = result.sort_values(["time", "grid_id"])
    return result


if __name__ == "__main__":
    all_dfs = []

    historical_dir = BASE_DIR / "data" / "raw" / "sewer" / "historical"
    zip_files = list(historical_dir.glob("*/*.zip"))

    if not zip_files:
        print("zip 파일을 찾지 못했습니다.")
        exit()

    for z in sorted(zip_files):
        if not z.exists():
            print(f"파일 없음, 건너뜀: {z}")
            continue

        print(f"\n처리 중: {z}")
        df = load_zip_file(str(z))

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("불러온 데이터가 없습니다.")
        exit()

    raw_result = pd.concat(all_dfs, ignore_index=True)
    raw_result = raw_result.sort_values("msrmt_ymd")

    final_result = attach_grid_and_transform(
        raw_result,
        str(BASE_DIR / "data" / "grid" / "sensor_grid_map.csv"),
    )

    output_dir = BASE_DIR / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "gangnam_sewer_historical_grid.csv"
    output_parquet = output_dir / "gangnam_sewer_historical_grid.parquet"

    final_result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    final_result.to_parquet(
        output_parquet,
        index=False,
    )

    print(final_result.head())
    print(f"총 {len(final_result)}건 저장 완료")
    print(f"CSV 저장 완료: {output_csv}")
    print(f"Parquet 저장 완료: {output_parquet}")