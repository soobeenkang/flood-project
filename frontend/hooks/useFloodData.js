import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { fetchHeatmap } from '../services/floodApi';
import useFloodStore from '../stores/floodStore';

/**
 * useFloodData
 *
 * - React Query로 /flood/heatmap 캐싱
 * - staleTime: 60초 (슬라이더 조작 시 캐시 우선 사용 → 부드러운 UX)
 * - 성공 시 floodStore.setFloodData 호출 → Canvas 재렌더 트리거
 *
 * @param {string} time - '', '1h', '3h', '6h'
 */
const useFloodData = (time = '') => {
  const setFloodData = useFloodStore((state) => state.setFloodData);

  const query = useQuery({
    queryKey: ['floodHeatmap', time],
    queryFn: () => fetchHeatmap(time),
    staleTime: 60 * 1000,       // 1분간 캐시 유지
    gcTime: 5 * 60 * 1000,      // 5분간 메모리 보존 (슬라이더 왔다갔다 대비)
    refetchOnWindowFocus: false,
  });

  // 데이터 fetch 성공 시 store 동기화
  useEffect(() => {
    if (query.data) {
      setFloodData(query.data);
    }
  }, [query.data, setFloodData]);

  return {
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
};

export default useFloodData;