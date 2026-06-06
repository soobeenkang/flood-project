import * as subscriptionsService from './subscriptions.service.js';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export async function createSubscription(req, res, next) {
  try {
    const { gridId, email } = req.body ?? {};

    if (!gridId || !email) {
      return res.status(400).json({
        code: 'LOCATION_REQUIRED',
        message: 'gridId와 email이 필요합니다.',
      });
    }
    if (!EMAIL_REGEX.test(email)) {
      return res.status(400).json({
        code: 'INVALID_EMAIL',
        message: '이메일 형식이 올바르지 않습니다.',
      });
    }
    if (!/^\d+$/.test(String(gridId))) {
      return res.status(400).json({
        code: 'INVALID_COORDINATES',
        message: '올바르지 않은 gridId 형식입니다.',
      });
    }

    const result = await subscriptionsService.createSubscription(String(gridId), email);
    return res.status(201).json(result);
  } catch (err) {
    if (err.code === 'DUPLICATE_SUBSCRIPTION') {
      return res.status(409).json({
        code: 'DUPLICATE_SUBSCRIPTION',
        message: '동일 gridId + 이메일이 이미 구독됩니다.',
      });
    }
    if (err.code === 'GRID_NOT_FOUND') {
      return res.status(404).json({
        code: 'GRID_NOT_FOUND',
        message: '해당 그리드를 찾을 수 없습니다.',
      });
    }
    next(err);
  }
}

export async function deleteSubscription(req, res, next) {
  try {
    const { subscriptionId } = req.params;

    if (!/^\d+$/.test(subscriptionId)) {
      return res.status(400).json({
        code: 'INVALID_COORDINATES',
        message: '올바르지 않은 subscriptionId 형식입니다.',
      });
    }

    const deleted = await subscriptionsService.deleteSubscription(subscriptionId);
    if (!deleted) {
      return res.status(404).json({
        code: 'GRID_NOT_FOUND',
        message: '해당 구독을 찾을 수 없습니다.',
      });
    }

    return res.json({ message: '구독이 해제되었습니다.' });
  } catch (err) {
    next(err);
  }
}