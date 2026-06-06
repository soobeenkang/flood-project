import axios from 'axios';
import https from 'https';
import redis from './redis.service.js';
import pool from "../db/pool.js";

const URL = 'https://www.safetydata.go.kr/V2/api/DSSP-IF-00247';

const httpsAgent = new https.Agent({
    rejectUnauthorized: false
});

export function initScheduler() {
    console.log('Alert scheduler started');

    collectAlerts();

    setInterval(async () => {
        await collectAlerts();
    }, 3 * 60 * 1000);
}
async function getWithRetry(url, config, retries = 3) {
    for (let i = 1; i <= retries; i++) {
        try {
            return await axios.get(url, {
                ...config,
                timeout: 10000
            });
            if (i > 1) {
                console.log(`재시도 ${i}회차 성공`);
            }
            
        } catch (err) {
            console.error(`API 호출 실패 (${i}/${retries})`, err.code);

            if (i === retries) {
                throw err;
            }

            await new Promise(resolve =>
                setTimeout(resolve, 2000)
            );
        }
    }
}

async function saveAlerts(alerts){
    for (const alert of alerts) {
        await pool.query(
            `
            INSERT INTO alerts (
                source_sn,
                region,
                type,
                level,
                message,
                issued_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (source_sn)
            DO NOTHING
            `,
            [
                alert.sourceSn,
                alert.region,
                alert.type,
                alert.level,
                alert.message,
                alert.issuedAt
            ]
        );
    }
}
async function collectAlerts() {
    try {
        const todayDate = new Date();

        const yesterdayDate = new Date(todayDate);
        yesterdayDate.setDate(yesterdayDate.getDate() - 1);

        const today = todayDate
            .toISOString()
            .slice(0, 10)
            .replace(/-/g, '');

        const yesterday = yesterdayDate
            .toISOString()
            .slice(0, 10)
            .replace(/-/g, '');

        // console.log('today 요청 시작');

        const todayResponse = await getWithRetry(URL, {
            httpsAgent,
            params: {
                serviceKey: process.env.A_SERVICE_KEY,
                returnType: 'json',
                pageNo: 1,
                numOfRows: 1000,
                crtDt: today,
                rgnNm: '서울'
            }
        });

        // console.log('today 요청 성공');

        // console.log('yesterday 요청 시작');

        // console.log('today=', today);
        // console.log('yesterday=', yesterday);
        // console.log("KEY:", process.env.A_SERVICE_KEY);
        // console.log("LEN:", process.env.A_SERVICE_KEY.length);
        const yesterdayResponse = await getWithRetry(URL, {
            httpsAgent,
            params: {
                serviceKey: process.env.A_SERVICE_KEY,
                returnType: 'json',
                pageNo: 1,
                numOfRows: 1000,
                crtDt: yesterday,
                rgnNm: '서울'
            }
        });
        
        // console.log('totalCount:', todayResponse.data.totalCount);
        // console.log('body length:', todayResponse.data.body?.length);

        const body = [
            ...(todayResponse.data.body ?? []),
            ...(yesterdayResponse.data.body ?? [])
        ];

        const alerts = body
            .filter(row =>
                ['호우', '홍수', '태풍'].includes(row.DST_SE_NM)
            )
            .map(row => ({
                sourceSn: row.SN,
                region: row.RCPTN_RGN_NM?.trim(),
                type: row.DST_SE_NM,
                level: row.EMRG_STEP_NM,
                message: row.MSG_CN,
                issuedAt: row.CRT_DT
            }));
        // console.log(
        //     alerts.map(a => a.sourceSn)
        // );

        if (alerts.length > 0){
            await saveAlerts(alerts);
            await redis.set(
                'alerts:latest',
                JSON.stringify(alerts)
            );
            console.log(`재난문자 ${alerts.length}건 저장`);
        } else {
            console.log('[스케줄러] 최근 24내 재난문자 데이터 없음.');
        }
        

    } catch (err) {
        console.error('재난문자 API 호출 실패');
        console.error(err);
    }
}

