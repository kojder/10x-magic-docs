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
        if (window.hljs) {
          const highlighted = window.hljs.highlight(code, { language }).value;
          setHighlightedCode(highlighted);
        }
      };
      langScript.onerror = () => {
        // Fallback to auto-detection if language file fails
        if (window.hljs) {
          const highlighted = window.hljs.highlightAuto(code).value;
          setHighlightedCode(highlighted);
        }
      };
      document.head.appendChild(langScript);
    } else {
      // Language already loaded
      if (window.hljs) {
        const highlighted = window.hljs.highlight(code, { language }).value;
        setHighlightedCode(highlighted);
      }
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
    <div className="bg-[#1e1e1e] rounded-lg border border-[#3e3e42] mb-6 overflow-hidden shadow-lg">
      {fileName && (
        <div className="bg-[#252526] px-4 py-2.5 border-b border-[#3e3e42] flex justify-between items-center">
          <span className="text-[#cccccc] font-mono text-sm">{fileName}</span>
          <button
            onClick={copyToClipboard}
            className="text-xs px-3 py-1.5 rounded bg-[#0e639c] hover:bg-[#1177bb] text-[#ffffff] transition-colors font-medium"
          >
            {copied ? "✓ Copied!" : "Copy"}
          </button>
        </div>
      )}
      <div className="relative overflow-auto bg-[#1e1e1e]">
        <pre className="p-4 overflow-x-auto flex m-0">
          {showLineNumbers && (
            <div className="text-[#858585] pr-4 text-right select-none border-r border-[#3e3e42] mr-4 min-w-[2.5rem]">
              {getLineNumbers()}
            </div>
          )}
          <code
            className={`language-${language} text-[#d4d4d4]`}
            dangerouslySetInnerHTML={{ __html: highlightedCode || code }}
          />
        </pre>
      </div>
    </div>
  );
};
