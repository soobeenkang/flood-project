/**
 * useWebSocket
 *
 * TODO: WebSocket flood_alert 구현 예정
 *
 * 구현 시 포함할 내용:
 * - exponential backoff 재연결 로직
 * - flood_alert payload 파싱 → alertStore 연동
 * - cleanup (컴포넌트 언마운트 시 ws.close())
 *
 * payload 구조 확정 후 작업 시작
 */

const useWebSocket = () => {
  // placeholder - 추후 구현
  return {
    isConnected: false,
    lastMessage: null,
  };
};

export default useWebSocket;