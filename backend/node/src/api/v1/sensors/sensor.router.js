import express from 'express';
import * as sensorController from "./sensor.controller.js";

const router = express.Router();

router.get(
    "/:sensorId",
    sensorController.getSensor
);

export default router;