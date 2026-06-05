import { Router } from 'express';
import subscriptionsController from './subscriptions.controller.js';

const router = Router();

// POST api/v1/subscriptions
router.post('/', subscriptionsController.createSubscription);

// DELETE api/v1/subscriptions/:subscriptionId
router.delete('/:subscriptionId', subscriptionsController.deleteSubscription);

export default router;