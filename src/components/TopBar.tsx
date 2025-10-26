import { Link } from "react-router-dom";
import { Input } from "./ui/input";
import { useApiKeyStore } from "../store/apiKey";
import { Button } from "./ui/button";
import { useState } from "react";

export default function TopBar() {
  const { apiKey, setApiKey } = useApiKeyStore();
  const [isEditing, setIsEditing] = useState(false);
  const [tempKey, setTempKey] = useState(apiKey);

  const handleSave = () => {
    setApiKey(tempKey);
    setIsEditing(false);
  };

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <h1 className="text-2xl font-semibold">
        <Link
          to="/"
          className="text-[#569cd6] hover:text-[#6ab0de] transition-colors flex items-center gap-2"
        >
          <span className="text-2xl">✨</span>
          <span className="text-[#4ec9b0]">Magic Docs by 10xDevs</span>
        </Link>
      </h1>

      <div className="flex flex-col gap-2 w-full sm:w-auto">
        {isEditing ? (
          <div className="flex gap-2 items-center">
            <Input
              type="password"
              placeholder="Enter your Anthropic API key"
              value={tempKey}
              onChange={(e) => setTempKey(e.target.value)}
              className="max-w-md bg-[#252526] border-[#3e3e42] text-[#d4d4d4] placeholder:text-[#858585]"
            />
            <Button onClick={handleSave} size="sm" className="bg-[#0e639c] hover:bg-[#1177bb]">
              Save
            </Button>
            <Button
              onClick={() => setIsEditing(false)}
              variant="outline"
              size="sm"
              className="border-[#3e3e42] text-[#d4d4d4] hover:bg-[#2a2d2e]"
            >
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex gap-2 items-center">
            <div className="text-sm text-[#858585]">API Key: {apiKey ? "••••••••" : "Not set"}</div>
            <Button
              onClick={() => setIsEditing(true)}
              variant="outline"
              size="sm"
              className="border-[#3e3e42] text-[#d4d4d4] hover:bg-[#2a2d2e]"
            >
              {apiKey ? "Change" : "Set API Key"}
            </Button>
          </div>
        )}
        <p className="text-xs text-[#858585]">
          The API key is being sent from your browser to the Anthropic API.
        </p>
      </div>
    </div>
  );
}
