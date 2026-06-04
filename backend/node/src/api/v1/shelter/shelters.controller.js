import * as sheltersService from './shelters.service.js';

export async function getShelters(req, res) {
  try {
    const result = await sheltersService.getShelters(req.query);
    return res.status(200).json(result);
  } catch (error) {
    return res.status(400).json({ message: error.message });
  }
}

export async function getShelterDetail(req, res) {
  try {
    const result = await sheltersService.getShelterDetail(req.params.shelterId);
    return res.status(200).json(result);
  } catch (error) {
    return res.status(404).json({ message: error.message });
  }
}