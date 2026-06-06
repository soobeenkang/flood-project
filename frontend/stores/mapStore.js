import { create } from 'zustand';
import { MAP_CONFIG } from '../constants/mapConfig';

const useMapStore = create((set) => ({
  // 카카오맵 인스턴스 (ref 대신 store에서 관리)
  mapInstance: null,

  // 현재 지도 중심 좌표
  center: MAP_CONFIG.center,

  // 현재 줌 레벨
  level: MAP_CONFIG.level,

  // 선택된 격자 grid_id (클릭 시 상세 패널)
  selectedGridId: null,

  setMapInstance: (instance) => set({ mapInstance: instance }),

  setCenter: (center) => set({ center }),

  setLevel: (level) => set({ level }),

  setSelectedGridId: (gridId) => set({ selectedGridId: gridId }),

  clearSelectedGrid: () => set({ selectedGridId: null }),
}));

export default useMapStore;