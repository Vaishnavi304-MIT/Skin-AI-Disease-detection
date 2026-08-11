import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101826",       // near-black, primary text
        slate: {
          950: "#0b1220",
          900: "#141d2e",
          800: "#1f2a3d",
          700: "#324058",
          600: "#4b5a75",
          500: "#6c7b96",
          400: "#95a2b8",
          300: "#c1c9d6",
          200: "#dfe3ea",
          100: "#eef1f5",
          50: "#f6f8fa",
        },
        clinic: {
          teal: "#0f7d78",     // primary accent — diagnostic / active states
          tealDark: "#0b5f5b",
          amber: "#b5762a",    // caution / confidence-mid
          red: "#a3352b",      // high-risk flag only
        },
      },
      fontFamily: {
        display: ["var(--font-source-serif)", "Georgia", "serif"],
        sans: ["var(--font-plex-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "5px",
        md: "6px",
        lg: "8px",
      },
      boxShadow: {
        panel: "0 1px 2px 0 rgb(16 24 38 / 0.06), 0 1px 12px -4px rgb(16 24 38 / 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
