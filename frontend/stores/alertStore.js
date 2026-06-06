import { create } from 'zustand';

let toastIdCounter = 0;

const useAlertStore = create((set) => ({
  // 토스트 알림 목록
  toasts: [],

  // 배너 알림 (상단 고정)
  banner: null,

  addToast: (toast) => {
    const id = ++toastIdCounter;
    set((state) => ({
      toasts: [...state.toasts, { id, ...toast }],
    }));
    // 5초 후 자동 제거
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, 5000);
    return id;
  },

  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  setBanner: (banner) => set({ banner }),
  clearBanner: () => set({ banner: null }),
}));

export default useAlertStore;