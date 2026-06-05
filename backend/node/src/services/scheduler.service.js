import axios from 'axios';
import https from 'https';
import redis from './redis.service.js';

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

        const todayResponse = await axios.get(URL, {
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
        const yesterdayResponse = await axios.get(URL, {
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
        // const body = todayResponse.data.body ?? [];

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
        
        await redis.set(
            'alerts:latest',
            JSON.stringify(alerts)
        );

        console.log(`재난문자 ${alerts.length}건 저장`);

    } catch (err) {
        console.error('재난문자 API 호출 실패');
        console.error(err);
    }
}
