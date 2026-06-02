import express from 'express';
import * as sensorController from "./sensor.controller.js";

const router = express.Router();
router.get(
    "/",
    sensorController.getSensors
);
router.post(
    "/ingest",
    sensorController.ingest
);
export default router;