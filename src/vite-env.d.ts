/// <reference types="vite/client" />

interface Window {
  hljs?: {
    highlight: (code: string, options: { language: string }) => { value: string };
    highlightAuto: (code: string) => { value: string };
    getLanguage: (name: string) => any;
  };
}
