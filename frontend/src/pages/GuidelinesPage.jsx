import { useState } from 'react';
import { GUIDELINES } from '../data/mockData';

const TABS = [
  { key: 'flood',      label: '침수',  emoji: '🌊' },
  { key: 'heavy_rain', label: '호우',  emoji: '🌧️' },
  { key: 'typhoon',    label: '태풍',  emoji: '🌀' },
];

const GuidelinesPage = () => {
  const [activeTab, setActiveTab]       = useState('flood');
  const [expandedId, setExpandedId]     = useState('01');

  const items = GUIDELINES[activeTab] ?? [];

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: '#F8FAFC', padding: '16px' }}>
      {/* 헤더 */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: '#9CA3AF', marginBottom: 4 }}>
          📖 행정안전부 · 국민재난안전포털
        </div>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#111' }}>재난 행동 지침</div>
        <div style={{ fontSize: 13, color: '#6B7280', marginTop: 4 }}>
          정부 공식 지침을 상황별로 정리했어요.
        </div>
      </div>

      {/* 탭 */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => { setActiveTab(tab.key); setExpandedId('01'); }}
            style={{
              padding: '10px 20px', border: 'none', borderRadius: 12,
              fontSize: 14, fontWeight: 700, cursor: 'pointer',
              background: activeTab === tab.key ? '#3B82F6' : 'white',
              color: activeTab === tab.key ? 'white' : '#374151',
              boxShadow: activeTab === tab.key
                ? '0 2px 8px rgba(59,130,246,0.3)'
                : '0 1px 3px rgba(0,0,0,0.08)',
            }}
          >
            {tab.emoji} {tab.label}
          </button>
        ))}
      </div>

      {/* 아코디언 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {items.map((item) => {
          const isOpen = expandedId === item.id;
          return (
            <div key={item.id} style={{
              background: 'white', borderRadius: 16,
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)', overflow: 'hidden',
            }}>
              <button
                onClick={() => setExpandedId(isOpen ? null : item.id)}
                style={{
                  width: '100%', padding: '16px 20px',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  border: 'none', background: 'transparent', cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: 8,
                    background: '#3B82F6', display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, fontWeight: 800, color: 'white',
                  }}>{item.id}</div>
                  <span style={{ fontSize: 16, fontWeight: 700, color: '#111' }}>{item.title}</span>
                </div>
                <span style={{ fontSize: 16, color: '#9CA3AF', transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
                  ‹
                </span>
              </button>

              {isOpen && (
                <div style={{ padding: '0 20px 16px' }}>
                  {item.items.map((text, i) => (
                    <div key={i} style={{
                      display: 'flex', gap: 10, padding: '8px 0',
                      borderTop: i > 0 ? '1px solid #F3F4F6' : 'none',
                    }}>
                      <div style={{
                        width: 6, height: 6, borderRadius: '50%',
                        background: '#3B82F6', flexShrink: 0, marginTop: 7,
                      }} />
                      <span style={{ fontSize: 14, color: '#374151', lineHeight: 1.6 }}>{text}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 출처 */}
      <div style={{ textAlign: 'center', padding: '24px 0 8px', fontSize: 12, color: '#9CA3AF' }}>
        출처 <span style={{ color: '#3B82F6' }}>안전디딤돌 · 국민재난안전포털 ↗</span>
      </div>
    </div>
  );
};

export default GuidelinesPage;
