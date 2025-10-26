import React from "react";
import { TextBlockProps } from "./tools.types";
import { useAnthropic } from "./hooks/useAnthropic";
import { ArrowUp, ArrowDown, Dumbbell } from "lucide-react";
import ReactMarkdown from "react-markdown";

const complexityLabels = ["Beginner", "Intermediate", "Professional", "Advanced", "Expert"];

export const TextBlock: React.FC<TextBlockProps> = ({ header, text }) => {
  const { modifyComplexity, isLoading, error } = useAnthropic();
  const [content, setContent] = React.useState(text);
  const [complexityLevel, setComplexityLevel] = React.useState<number>(2); // Start at Professional level
  const [contentCache, setContentCache] = React.useState<Record<number, string>>({
    2: text, // Initialize cache with the original text at level 2 (Professional)
  });

  // Process content to ensure proper markdown formatting
  const processedContent = React.useMemo(() => {
    // Replace escaped newlines with actual newlines for markdown
    return content.replace(/\\n/g, "\n");
  }, [content]);

  const handleComplexityChange = async (action: "increase" | "decrease") => {
    // Calculate new complexity level
    const newComplexityLevel =
      action === "increase" ? Math.min(complexityLevel + 1, 4) : Math.max(complexityLevel - 1, 0);

    // Don't proceed if we're already at min/max level
    if (newComplexityLevel === complexityLevel) return;

    // Check if we already have this complexity level in cache
    if (contentCache[newComplexityLevel]) {
      setContent(contentCache[newComplexityLevel]);
      setComplexityLevel(newComplexityLevel);
      return;
    }

    try {
      const newContent = await modifyComplexity(content, header, action, complexityLevel);

      // Update content and cache
      setContent(newContent);
      setContentCache((prev) => ({
        ...prev,
        [newComplexityLevel]: newContent,
      }));
      setComplexityLevel(newComplexityLevel);
    } catch (err) {
      console.error("Failed to modify content:", err);
    }
  };

  return (
    <div className="bg-[#1e1e1e] rounded-lg border border-[#3e3e42] p-6 mb-6 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-[#4ec9b0]">{header}</h2>
        </div>
        <div className="flex gap-2 items-center">
          <span className="flex flex-row items-center gap-2 text-sm text-[#858585]">
            <Dumbbell size={16} />
            <span>{complexityLabels[complexityLevel]}</span>
          </span>

          <button
            onClick={() => handleComplexityChange("increase")}
            disabled={isLoading || complexityLevel >= 4}
            className="px-2 py-1 text-sm rounded bg-[#0e639c] text-[#e0e0e0] hover:bg-[#1177bb] disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 transition-colors"
          >
            <ArrowUp size={16} />
          </button>
          <button
            onClick={() => handleComplexityChange("decrease")}
            disabled={isLoading || complexityLevel <= 0}
            className="px-2 py-1 text-sm rounded bg-[#0e639c] text-[#e0e0e0] hover:bg-[#1177bb] disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 transition-colors"
          >
            <ArrowDown size={16} />
          </button>
        </div>
      </div>
      <div className="text-[#d4d4d4] leading-relaxed relative prose prose-invert max-w-none">
        {isLoading && (
          <div className="absolute inset-0 bg-[#1e1e1e]/80 flex items-center justify-center rounded-lg">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#4ec9b0]"></div>
          </div>
        )}
        {error && <div className="text-[#f48771] mb-2 text-sm">Error: {error.message}</div>}
        <ReactMarkdown
          components={{
            p: ({ children }) => <p className="mb-4">{children}</p>,
            strong: ({ children }) => (
              <strong className="font-semibold text-[#4ec9b0]">{children}</strong>
            ),
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>
    </div>
  );
};
