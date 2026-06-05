// api/v1/heatmap/heatmap.router.js
import { Router } from 'express';
import { getGrids, getGrid, getAllHorizons } from './heatmap.controller.js';

const router = Router();

// 특정 시간대 히트맵 그리드 목록
// GET /api/v1/heatmap/grids?lat=&lon=&radius=&horizon=now|1h|3h|6h
router.get('/grids', getGrids);

// 한 번에 모든 시간대 데이터 (프론트 버튼 전환 대응)
// GET /api/v1/heatmap/grids/all-horizons?lat=&lon=&radius=
router.get('/grids/all-horizons', getAllHorizons);

// 특정 그리드 상세 (시간대별 depth 포함)
// GET /api/v1/heatmap/grids/:id
router.get('/grids/:id', getGrid);

export default router;