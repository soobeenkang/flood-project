import crypto from 'crypto';
import pool from '../../../db/pool.js';

const ALGO = 'aes-256-cbc';
// EMAIL_ENCRYPT_KEY: 64자 hex 문자열 → 32바이트 Buffer
const KEY = Buffer.from(
  (process.env.EMAIL_ENCRYPT_KEY || '').padEnd(64, '0').slice(0, 64),
  'hex',
);

// ── 암호화 헬퍼 ──────────────────────────────────────────────

function encryptEmail(email) {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(ALGO, KEY, iv);
  const encrypted = Buffer.concat([cipher.update(email, 'utf8'), cipher.final()]);
  return `${iv.toString('hex')}:${encrypted.toString('hex')}`;
}

function decryptEmail(encrypted) {
  const [ivHex, ctHex] = encrypted.split(':');
  const decipher = crypto.createDecipheriv(ALGO, KEY, Buffer.from(ivHex, 'hex'));
  return Buffer.concat([decipher.update(Buffer.from(ctHex, 'hex')), decipher.final()]).toString('utf8');
}

/** SHA-256(gridId:email) — 중복 체크용 해시 */
function emailHash(gridId, email) {
  return crypto
    .createHash('sha256')
    .update(`${gridId}:${email.toLowerCase()}`)
    .digest('hex');
}

function maskEmail(email) {
  const [local, domain] = email.split('@');
  return `${local.slice(0, 2)}***@${domain}`;
}

// ── 서비스 함수 ──────────────────────────────────────────────

/**
 * 구독 등록.
 * @throws {{ code: 'GRID_NOT_FOUND' | 'DUPLICATE_SUBSCRIPTION' }}
 */
export async function createSubscription(gridId, email) {
  const { rows: gridRows } = await pool.query(
    'SELECT grid_id FROM flood_grid WHERE grid_id = $1',
    [BigInt(gridId)],
  );
  if (gridRows.length === 0) {
    const err = new Error('Grid not found');
    err.code = 'GRID_NOT_FOUND';
    throw err;
  }

  try {
    const { rows } = await pool.query(
      `INSERT INTO subscriptions (grid_id, email_encrypted, email_hash, created_at)
       VALUES ($1, $2, $3, NOW())
       RETURNING subscription_id, grid_id, created_at`,
      [BigInt(gridId), encryptEmail(email), emailHash(gridId, email)],
    );
    const row = rows[0];
    return {
      subscriptionId: String(row.subscription_id),
      gridId: String(row.grid_id),
      email: maskEmail(email),
      createdAt: row.created_at,
    };
  } catch (err) {
    if (err.code === '23505') {
      const dupErr = new Error('Duplicate subscription');
      dupErr.code = 'DUPLICATE_SUBSCRIPTION';
      throw dupErr;
    }
    throw err;
  }
}

/**
 * 구독 해제.
 * @returns {Promise<boolean>}
 */
export async function deleteSubscription(subscriptionId) {
  const { rowCount } = await pool.query(
    'DELETE FROM subscriptions WHERE subscription_id = $1',
    [BigInt(subscriptionId)],
  );
  return rowCount > 0;
}