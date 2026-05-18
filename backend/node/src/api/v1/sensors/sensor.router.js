import express from 'express';

const router = express.Router();

router.get('/', (req, res) => {
    res.json({
        message: 'sensor route ok'
    });
});

export default router;