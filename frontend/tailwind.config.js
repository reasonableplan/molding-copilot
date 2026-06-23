/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#18181b',        // near-black 텍스트/버튼
        accent: '#c4ec3f',     // 라임 — 브랜드/활성 크롬 (긍정·하이라이트)
        accentdk: '#a9d62a',
        gas: '#ef4444',        // 과열·가스 (의미색)
        short: '#2563eb',      // 충전·미성형
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif'],
      },
      borderRadius: {
        card: '22px',
      },
      boxShadow: {
        card: '0 1px 2px rgba(0,0,0,0.04), 0 8px 24px -12px rgba(0,0,0,0.10)',
      },
    },
  },
  plugins: [],
}
