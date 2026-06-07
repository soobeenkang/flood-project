import express from 'express';
import { getWeatherByLocation } from './weather.controller.js';

const router = express.Router();

router.get('/', getWeatherByLocation);

export default router;