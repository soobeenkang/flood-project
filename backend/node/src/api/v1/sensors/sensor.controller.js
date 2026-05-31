import * as sensorService from "./sensor.service.js";

export async function getSensor(req,res){
    const sensorId = req.params.sensorId;

    const result = 
        await sensorService.getSensor(sensorId);
    
        if(!result){

            return res.status(404).json({
                message : "센서 없음"
            });
        }
        
        res.json(result);
}