import React, { useState } from "react";
import { QuizProps } from "./tools.types";

export const Quiz: React.FC<QuizProps> = ({ title, question }) => {
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleOptionSelect = (optionId: string) => {
    setSelectedAnswer(optionId);
  };

  const handleCheckAnswer = () => {
    if (selectedAnswer) {
      setShowExplanation(true);
    }
  };

  const handleReset = () => {
    setSelectedAnswer(null);
    setShowExplanation(false);
  };

  const isCorrect = selectedAnswer === question.correctAnswer;

  return (
    <div className="bg-[#1e1e1e] rounded-lg border border-[#3e3e42] p-6 mb-6 shadow-lg">
      {title && <h3 className="text-lg font-semibold mb-4 text-[#569cd6]">{title}</h3>}

      <div className="mb-4">
        <p className="text-[#d4d4d4] font-medium mb-4">{question.question}</p>

        <div className="space-y-2.5">
          {question.options.map((option) => (
            <button
              key={option.id}
              onClick={() => handleOptionSelect(option.id)}
              disabled={showExplanation}
              className={`w-full text-left p-3.5 rounded border transition-all ${
                selectedAnswer === option.id
                  ? showExplanation
                    ? isCorrect
                      ? "bg-[#1e3a1e] border-[#4ec9b0] text-[#d4d4d4]"
                      : "bg-[#3a1e1e] border-[#f48771] text-[#d4d4d4]"
                    : "bg-[#264f78] border-[#569cd6] text-[#ffffff]"
                  : "bg-[#252526] border-[#3e3e42] hover:border-[#569cd6] hover:bg-[#2a2d2e] text-[#d4d4d4]"
              }`}
            >
              <span className="font-bold mr-2 text-[#4ec9b0]">{option.id}.</span> {option.text}
            </button>
          ))}
        </div>
      </div>

      {!showExplanation ? (
        <button
          onClick={handleCheckAnswer}
          disabled={!selectedAnswer}
          className={`px-5 py-2.5 rounded font-medium transition-colors ${
            selectedAnswer
              ? "bg-[#0e639c] hover:bg-[#1177bb] text-[#ffffff]"
              : "bg-[#3e3e42] text-[#858585] cursor-not-allowed"
          }`}
        >
          Check Answer
        </button>
      ) : (
        <div>
          <div
            className={`mt-4 p-4 rounded border ${
              isCorrect ? "bg-[#1e3a1e] border-[#4ec9b0]" : "bg-[#3a1e1e] border-[#f48771]"
            }`}
          >
            <p
              className={`font-semibold text-lg ${isCorrect ? "text-[#4ec9b0]" : "text-[#f48771]"}`}
            >
              {isCorrect ? "✓ Correct!" : "✗ Incorrect!"}
            </p>
            {question.explanation && (
              <p className="text-[#d4d4d4] mt-2 leading-relaxed">{question.explanation}</p>
            )}
          </div>

          <button
            onClick={handleReset}
            className="mt-4 px-5 py-2.5 bg-[#3e3e42] text-[#cccccc] rounded hover:bg-[#505050] transition-colors font-medium"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};
