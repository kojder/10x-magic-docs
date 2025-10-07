import React, { useEffect, useState } from "react";
import { CodeSnippetProps } from "./tools.types";

export const CodeSnippet: React.FC<CodeSnippetProps> = ({
  code,
  language = "typescript",
  fileName,
  showLineNumbers = true,
}) => {
  const [highlightedCode, setHighlightedCode] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);
  const [hljsLoaded, setHljsLoaded] = useState<boolean>(false);

  useEffect(() => {
    // Load highlight.js from CDN if not already loaded
    if (!window.hljs) {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js";
      script.onload = () => setHljsLoaded(true);
      document.head.appendChild(script);

      // Load CSS for syntax highlighting
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href =
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css";
      document.head.appendChild(link);
    } else {
      setHljsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!hljsLoaded || !window.hljs) return;

    // Load the specific language if not already loaded
    if (!window.hljs.getLanguage(language)) {
      const langScript = document.createElement("script");
      langScript.src = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/languages/${language}.min.js`;
      langScript.onload = () => {
        const highlighted = window.hljs.highlight(code, { language }).value;
        setHighlightedCode(highlighted);
      };
      langScript.onerror = () => {
        // Fallback to auto-detection if language file fails
        const highlighted = window.hljs.highlightAuto(code).value;
        setHighlightedCode(highlighted);
      };
      document.head.appendChild(langScript);
    } else {
      // Language already loaded
      const highlighted = window.hljs.highlight(code, { language }).value;
      setHighlightedCode(highlighted);
    }
  }, [code, language, hljsLoaded]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLineNumbers = () => {
    const lines = code.split("\n").length;
    return Array.from({ length: lines }, (_, i) => i + 1).join("\n");
  };

  return (
    <div className="bg-[#1a1a1a] rounded-lg border border-gray-800 mb-6 overflow-hidden">
      {fileName && (
        <div className="bg-[#242424] px-4 py-2 border-b border-gray-800 flex justify-between items-center">
          <span className="text-gray-300 font-mono text-sm">{fileName}</span>
          <button
            onClick={copyToClipboard}
            className="text-xs px-2 py-1 rounded bg-blue-900 hover:bg-blue-800 text-blue-200 transition-colors"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      )}
      <div className="relative overflow-auto">
        <pre className="p-4 overflow-x-auto flex">
          {showLineNumbers && (
            <div className="text-gray-500 pr-4 text-right select-none border-r border-gray-700 mr-4">
              {getLineNumbers()}
            </div>
          )}
          <code
            className={`language-${language}`}
            dangerouslySetInnerHTML={{ __html: highlightedCode || code }}
          />
        </pre>
      </div>
    </div>
  );
};
