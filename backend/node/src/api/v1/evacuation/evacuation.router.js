import { Router } from 'express';
import { getEvacuationRoute } from './evacuation.controller.js';
 
const router = Router();
 
// 대피 경로 탐색
// GET /api/v1/route/evacuation?startLat=&startLon=&endLat=&endLon=
router.get('/evacuation', getEvacuationRoute);
 
export default router;