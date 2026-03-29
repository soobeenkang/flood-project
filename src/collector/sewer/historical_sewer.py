import zipfile
from pathlib import Path
import pandas as pd

# 프로젝트 루트 기준 경로
BASE_DIR = Path(__file__).resolve().parents[3]

# 서울 전체 센서 기준 grid map 파일
GRID_MAP_PATH = BASE_DIR / "data" / "grid" / "sensor_grid_map.csv"

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


def read_csv_fast(file_obj, **kwargs) -> pd.DataFrame:
    """
    가능한 한 빠르게 CSV를 읽는다.
    우선 c 엔진 + utf-8 시도
    실패하면 c 엔진 + cp949
    그래도 실패하면 python 엔진 + cp949
    """
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, engine="c", encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, engine="c", encoding="cp949", **kwargs)
        except Exception:
            file_obj.seek(0)
            return pd.read_csv(file_obj, engine="python", encoding="cp949", **kwargs)
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, engine="python", encoding="cp949", **kwargs)


def standardize_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    다양한 헤더명을 표준 컬럼명으로 통일하고,
    부족한 컬럼은 None으로 채운 뒤 STANDARD_COLUMNS 순서로 반환
    """
    df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["source_file"] = source_file
    return df[STANDARD_COLUMNS]


def read_old_csv_file(file_obj, source_file: str) -> pd.DataFrame:
    """
    헤더 없는 옛날 CSV 파일 읽기 (2012~2018)
    """
    old_cols = [
        "unq_no",
        "legacy_se_cd",
        "se_nm",
        "legacy_time",
        "msrmt_ymd",
        "msrmt_watl",
        "sgn_stts",
    ]

    df = read_csv_fast(
        file_obj,
        header=None,
        names=old_cols,
        on_bad_lines="skip",
    )

    # 표준 컬럼 맞추기
    df["se_cd"] = df["legacy_se_cd"]
    df["pstn_info"] = None

    return standardize_columns(df, source_file)


def read_csv_file(file_obj, source_file: str) -> pd.DataFrame:
    """
    헤더 있는 최근 CSV 파일 읽기 (2019~)
    """
    df = read_csv_fast(
        file_obj,
        on_bad_lines="skip",
    )

    return standardize_columns(df, source_file)


def clean_types(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    자료형 정리
    """
    df = df.copy()

    df["msrmt_watl"] = pd.to_numeric(df["msrmt_watl"], errors="coerce")

    # 구형 데이터(2012~2018): 예) 01-MAY-12 00:00:00
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


def load_zip_file(zip_path: str) -> pd.DataFrame:
    """
    zip 안에 있는 CSV 파일들을 읽어서 하나의 DataFrame으로 반환
    """
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


def attach_grid_and_transform(
    df: pd.DataFrame,
    grid_map_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    센서 데이터에 grid_id를 붙이고 모델 입력용 형태로 변환
    최종 컬럼: time, grid_id, water_level
    + 매핑 안 된 센서 목록 반환
    """

    df = df.copy()
    df["unq_no"] = df["unq_no"].astype(str).str.strip()

    # grid 붙이기
    merged = df.merge(
        grid_map_df[["sensor_id", "grid_id"]],
        left_on="unq_no",
        right_on="sensor_id",
        how="left",
    )
    """
    # 매핑 안 된 센서 확인
    unmatched = (
        merged[merged["grid_id"].isna()][["unq_no", "se_cd", "se_nm", "pstn_info"]]
        .drop_duplicates()
        .copy()
    )

    if not unmatched.empty:
        print("\n매핑 안 된 센서 수:", unmatched["unq_no"].nunique())
        print(unmatched.head(30))

    """
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
    result["water_level"] = pd.to_numeric(result["water_level"], errors="coerce").astype("float32")

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
    result = result.sort_values(["time", "grid_id"]).reset_index(drop=True)
    return result

if __name__ == "__main__":
    historical_dir = BASE_DIR / "data" / "raw" / "sewer" / "historical"
    zip_files = list(historical_dir.glob("*/*.zip"))

    if not zip_files:
        print("zip 파일을 찾지 못했습니다.")
        raise SystemExit

    output_dir = BASE_DIR / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "seoul_sewer_historical_grid.csv"
    output_parquet = output_dir / "seoul_sewer_historical_grid.parquet"
    unmatched_csv = output_dir / "historical_unmatched_sensors.csv"

    # 기존 결과 삭제하고 새로 만들고 싶으면 주석 해제
    if output_csv.exists():
        output_csv.unlink()
    if output_parquet.exists():
        output_parquet.unlink()
    if unmatched_csv.exists():
        unmatched_csv.unlink()

    grid_map_df = pd.read_csv(GRID_MAP_PATH)
    grid_map_df["sensor_id"] = grid_map_df["sensor_id"].astype(str).str.strip()

    total_rows = 0
    all_unmatched = []

    for z in sorted(zip_files):
        if not z.exists():
            print(f"파일 없음, 건너뜀: {z}")
            continue

        print(f"\n처리 중: {z}")
        df = load_zip_file(str(z))

        if df.empty:
            print("비어 있어서 건너뜀")
            continue

        transformed = attach_grid_and_transform(df, grid_map_df)
        """
        if not unmatched.empty:
            all_unmatched.append(unmatched)
        """

        if transformed.empty:
            print("변환 결과가 비어서 건너뜀")
            continue

        # 기존 파일 있으면 읽어서 합치기
        if output_parquet.exists():
            existing = pd.read_parquet(output_parquet)
            combined = pd.concat([existing, transformed], ignore_index=True)
        else:
            combined = transformed

        # 같은 시간 + 같은 grid는 최대 수위만 남기기
        combined = (
            combined.groupby(["time", "grid_id"], as_index=False)["water_level"]
            .max()
            .sort_values(["time", "grid_id"])
            .reset_index(drop=True)
        )

        combined.to_parquet(output_parquet, index=False)
        combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

        total_rows = len(combined)
        print(f"중간 저장 완료: {z.name}")
        print(f"현재 누적 행 수: {total_rows}")

    """
    # unmatched는 비어도 파일 생성
    if all_unmatched:
        unmatched_total = pd.concat(all_unmatched, ignore_index=True)
        unmatched_total = unmatched_total.drop_duplicates(subset=["unq_no"]).reset_index(drop=True)
    else:
        unmatched_total = pd.DataFrame(columns=["unq_no", "se_cd", "se_nm", "pstn_info"])

    unmatched_total.to_csv(unmatched_csv, index=False, encoding="utf-8-sig")
    print(f"미매핑 센서 저장 완료: {unmatched_csv}")
    print(f"전체 누적 미매핑 센서 수: {len(unmatched_total)}")

    if not output_parquet.exists():
        print("최종 저장된 데이터가 없습니다.")
        raise SystemExit
    """

    final_result = pd.read_parquet(output_parquet)

    print("\n===== 저장 결과 =====")
    print(final_result.head())
    print(final_result.dtypes)
    print(f"총 {len(final_result)}건 저장 완료")
    print(f"CSV 저장 완료: {output_csv}")
    print(f"Parquet 저장 완료: {output_parquet}")