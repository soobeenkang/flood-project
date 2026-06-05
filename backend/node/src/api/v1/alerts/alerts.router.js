import { Router } from 'express';
import { list } from './alerts.controller.js';

const router = Router();

// GET /api/v1/alerts?lat=&lon=&limit=
router.get('/', list);

export default router;