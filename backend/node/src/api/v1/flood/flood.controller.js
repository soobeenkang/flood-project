import * as service from './flood.service.js';

export const getHeatmap = async (req, res, next) => {
    try {
        const horizon = req.query.t || 'current';
        const data = await service.getHeatmapByHorizon(horizon);
        res.json(data);
    } catch (e) { 
        next(e); 
    }
};

export const getGrid = async (req, res, next) => {
    try {
        const { gridId } = req.params;
        const data = await service.getGridDetails(gridId);

        if (!data) {
            return res.status(404).json({
                status: 'error',
                message: `해당 격자: ${gridId} 정보 찾을 수 없음.` 
            });
        }
        res.json(data);
    } catch (e) { 
        next(e); 
    }
};