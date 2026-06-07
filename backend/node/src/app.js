//FIXME
import express from 'express';
import cors from 'cors';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

// 라우터
import sensorRouter from './api/v1/sensors/sensor.router.js';
import sheltersRouter from './api/v1/shelter/shelters.router.js';
import heatmapRouter from './api/v1/heatmap/heatmap.router.js';
import locationRouter from './api/v1/location/location.router.js';
import alertsRouter from './api/v1/alerts/alerts.router.js';
import subscriptionsRouter from './api/v1/subscriptions/subscriptions.router.js';
import evacuationRouter from './api/v1/evacuation/evacuation.router.js';
import weatherRouter from './api/v1/weather/weather.router.js';
// 미구현 라우터 주석
// import adminRouter from './api/v1/admin/admin.router.js';

import errorHandler from './middlewares/errorHandler.js';
import { initMqtt } from './services/mqtt.service.js';
import { initWss } from './socket/wsManager.js';
import { initScheduler } from './services/scheduler.service.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(cors());
app.use(express.json());


// API 라우터 마운트
app.use('/api/v1/sensors', sensorRouter);
app.use('/api/v1/shelters', sheltersRouter);
app.use('/api/v1/heatmap', heatmapRouter);
app.use('/api/v1/location', locationRouter);
app.use('/api/v1/alerts', alertsRouter);
app.use('/api/v1/subscriptions', subscriptionsRouter);
app.use('/api/v1/route', evacuationRouter);
app.use('/api/v1/weather', weatherRouter);

/*
// 경로 수정 버전
app.use('/sensors', sensorRouter);
app.use('/shelters', sheltersRouter);
app.use('/heatmap', heatmapRouter);
app.use('/location', locationRouter);
app.use('/alerts', alertsRouter);
app.use('/subscriptions', subscriptionsRouter);
app.use('/route', evacuationRouter);
*/
app.use(express.static(path.join(__dirname, '../frontend/dist')));

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
});

//app.use('/api/v1/admin', adminRouter);

// 공통 에러 핸들러
app.use(errorHandler);

const server = http.createServer(app);

// 인프라 초기화
initMqtt();   // 아두이노 mqtt 구독 시작
initWss(server);
initScheduler();  // cron 작업 등록

server.listen(3000, () => console.log('API listening on :3000'));