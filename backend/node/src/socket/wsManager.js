const { WebSocketServer } = require('ws');

// channel → Set<WebSocket>
const channels = {
  alerts: new Set(),
  heatmap: new Set(),
};

let wss = null;

/**
 * WebSocket 서버 초기화. app.js 에서 httpServer 생성 후 호출.
 * @param {import('http').Server} httpServer
 */
function initWss(httpServer) {
  wss = new WebSocketServer({ server: httpServer, path: '/ws' });

  wss.on('connection', (ws, req) => {
    console.log(`[WS] New connection from ${req.socket.remoteAddress}`);

    ws._aliveTimer = setTimeout(() => {
      console.log('[WS] Closing idle connection');
      ws.terminate();
    }, 60_000);

    ws.on('message', (raw) => {
      clearTimeout(ws._aliveTimer);
      ws._aliveTimer = setTimeout(() => ws.terminate(), 60_000);

      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return ws.send(JSON.stringify({ error: 'Invalid JSON' }));
      }

      if (msg.action === 'ping') {
        return ws.send(JSON.stringify({ action: 'pong', timestamp: new Date().toISOString() }));
      }

      if (msg.action === 'subscribe' && Array.isArray(msg.channels)) {
        msg.channels.forEach((ch) => {
          if (channels[ch]) channels[ch].add(ws);
        });
        ws.send(JSON.stringify({ action: 'subscribed', channels: msg.channels }));
      }
    });

    ws.on('close', () => {
      clearTimeout(ws._aliveTimer);
      Object.values(channels).forEach((s) => s.delete(ws));
    });

    ws.on('error', (err) => console.error('[WS] Socket error:', err.message));
  });

  console.log('[WS] WebSocket server ready at /ws');
}

/**
 * 특정 채널에 메시지 브로드캐스트.
 * @param {'alerts'|'heatmap'} channel
 * @param {object} data
 */
function broadcast(channel, data) {
  const set = channels[channel];
  if (!set || set.size === 0) return;

  const payload = JSON.stringify({ channel, data });
  set.forEach((ws) => {
    if (ws.readyState === ws.OPEN) ws.send(payload);
    else set.delete(ws);
  });
}

module.exports = { initWss, broadcast };