import { create } from 'zustand';

/**
 * floodStore
 *
 * floodMap: Map<grid_id, flood>
 *   - GeoJSON 좌표는 FloodHeatmap 컴포넌트가 직접 들고 있음
 *   - 여기서는 침수 예측값만 관리 (색상 lookup용)
 *
 * selectedTime: 현재 선택된 시간 슬라이더 값
 */
const useFloodStore = create((set) => ({
  // Map<number, number>  key: grid_id, value: flood(0|1)
  floodMap: new Map(),

  // 선택된 시간 ('', '1h', '3h', '6h')
  selectedTime: '',

  // 히트맵 데이터 배열 → Map으로 변환하여 저장
  setFloodData: (dataArray) => {
    const map = new Map();
    dataArray.forEach(({ grid_id, flood }) => {
      map.set(grid_id, flood);
    });
    set({ floodMap: map });
  },

  setSelectedTime: (time) => set({ selectedTime: time }),

  clearFloodData: () => set({ floodMap: new Map() }),
}));

export default useFloodStore;