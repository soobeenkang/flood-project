import * as sensorService from "./sensor.service.js";

export async function getSensors(req, res){

    console.log("GET /sensors 요청");

    const sensors =
        await sensorService.getSensors();

    res.json({
        sensors
    });
}
export async function ingest(req, res) {

    console.log("POST /sensors/ingest 요청");

    const result =
        await sensorService.ingest(req.body);

    res.json(result);
}