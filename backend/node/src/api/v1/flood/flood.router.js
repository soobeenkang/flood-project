import { Router } from 'express';
import * as ctrl from './flood.controller.js';

const router = Router();
router.get('/heatmap', ctrl.getHeatmap);    // ?t=current|1h|3h|6h
router.get('/grid/:gridId', ctrl.getGrid);  // 특정 격자 상세

export default router;