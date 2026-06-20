import glob
import os
import re
from pathlib import Path
import pandas as pd
import config

def filter_all_caches_with_cleaning():
    cache_dir = config.CACHE_DIR
    prefix = config.CACHE_PREFIX
    rain_threshold = getattr(config, "RAIN_FILTER_THRESHOLD", 5.0)

    search_pattern = os.path.join(cache_dir, f"{prefix}*.parquet")
    cache_files = sorted(glob.glob(search_pattern))

    if not cache_files:
        print(f"[오류] 캐시 디렉토리({cache_dir})에서 파일을 찾을 수 없습니다.")
        return

    print("🚨 [노이즈 정제 및 강수량 필터링 시작]")
    print("-" * 60)

    for file_path in cache_files:
        filename = Path(file_path).name
        year_match = re.search(r"(\d{4})", filename)
        year = year_match.group(1) if year_match else filename

        print(f"[{year}년 데이터 처리 중] {filename} 로드...")
        df = pd.read_parquet(file_path)

        # 필수 컬럼 체크
        if not all(col in df.columns for col in ["time", "flood", "rain_1h"]):
            print(f" ❌ [패스] 필수 컬럼이 없어 건너뜁니다.")
            print("-" * 60)
            continue

        init_rows = len(df)

        # ── 1단계: 시계열 선행 노이즈 필터링 ──
        # 비가 전혀 안 오는데(rain_1h == 0) 침수(flood == 1)라고 찍힌 잘못된 행(노이즈)을 찾습니다.
        noise_cond = (df["rain_1h"] == 0) & (df["flood"] == 1)
        df_cleaned = df[~noise_cond].reset_index(drop=True)
        
        noise_count = init_rows - len(df_cleaned)

        # ── 2단계: 요청하신 시간대별 강수량 필터링 ──
        # (정제된 데이터 기준) 동일 시간대(time)별로 침수 발생 총합 계산
        flood_by_time = df_cleaned.groupby("time")["flood"].transform("sum")

        # 조건 A: 해당 시간대에 정제된 침수 건수가 1건이라도 있음 (유지)
        cond_has_flood = flood_by_time > 0
        # 조건 B: 해당 시간대에 침수가 전혀 없지만, 1시간 강수량이 기준치 이상임
        cond_no_flood_but_heavy_rain = (flood_by_time == 0) & (df_cleaned["rain_1h"] >= rain_threshold)

        # 최종 필터링 적용
        df_final = df_cleaned[cond_has_flood | cond_no_flood_but_heavy_rain].reset_index(drop=True)
        final_rows = len(df_final)

        # ── 결과 출력 및 저장 ──
        print(f"  📊 초기 데이터 수    : {init_rows:,}")
        print(f"  ✂️  1단계 (노이즈 제거): -{noise_count:,} 행 (비 없이 침수된 데이터)")
        print(f"  ✂️  2단계 (강수량 필터): -{len(df_cleaned) - final_rows:,} 행 (침수 없는 시간대 rain_1h < {rain_threshold}mm)")
        print(f"  ✅ 최종 남은 데이터  : {final_rows:,} 행")

        # 원본 캐시 파일에 덮어쓰기
        df_final.to_parquet(file_path, index=False)
        print(f"  💾 저장 완료 → {filename}")
        print("-" * 60)

if __name__ == "__main__":
    # ⚠️ 중요: 데이터가 깎여나가므로 실행 전 기존 캐시 폴더 백업을 강력 권장합니다.
    filter_all_caches_with_cleaning()