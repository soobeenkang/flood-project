import express from 'express';
import {
  getShelters,
  getShelterDetail,
} from './shelters.controller.js';

const router = express.Router();

// 주변 대피소 목록 조회
// GET /api/v1/shelters?lat=35.8714&lon=128.6014&radius=3000
router.get('/', getShelters);

// 대피소 상세 조회
// GET /api/v1/shelters/:shelterId
router.get('/:shelterId', getShelterDetail);

export default router;