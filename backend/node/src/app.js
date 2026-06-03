import express from 'express';
import cors from 'cors';
import http from 'http';

import sensorRouter from './api/v1/sensors/sensor.router.js';
import floodRouter from './api/v1/flood/flood.router.js';
import sheltersRouter from './api/v1/shelter/shelters.router.js';

// 미구현 라우터 주석
// import evacuationRouter from './api/v1/evacuation/evacuation.router.js';
// import adminRouter from './api/v1/admin/admin.router.js';

import errorHandler from './middlewares/errorHandler.js';
import { initMqtt } from './services/mqtt.service.js';
// import { initScheduler } from './services/scheduler.service.js';

const app = express();

app.use(cors());
app.use(express.json());

// API 라우터 등록
app.use('/api/v1/sensors', sensorRouter);
app.use('/api/v1/flood', floodRouter);
app.use('/api/v1/shelters', sheltersRouter);

/* TODO */
// app.use('/api/v1/evacuation', evacuationRouter);
// app.use('/api/v1/admin', adminRouter);

// 공통 에러 핸들러
app.use(errorHandler);

const server = http.createServer(app);

// 인프라 초기화
initMqtt(); // 아두이노 mqtt 구독 시작
// initScheduler(); // cron 작업 등록

server.listen(3000, () => console.log('API listening on :3000'));