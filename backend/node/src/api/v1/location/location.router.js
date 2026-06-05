import { Router } from 'express';
import { floodCheck } from './location.controller.js';

const router = Router();

// GET api/v1/location/flood-check?lat=&lon=
router.get('/flood-check', floodCheck);

export default router;