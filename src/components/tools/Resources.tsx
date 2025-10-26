import React from "react";
import { ResourcesProps } from "./tools.types";
import { Link as LinkIcon } from "lucide-react";

export const Resources: React.FC<ResourcesProps> = ({ title, links }) => {
  return (
    <div className="bg-[#1e1e1e] rounded-lg border border-[#3e3e42] p-6 mb-6 shadow-lg">
      <h2 className="text-xl font-semibold mb-5 text-[#4ec9b0]">{title}</h2>

      <ul className="space-y-4">
        {links.map((link, index) => (
          <li key={index} className="border-b border-[#3e3e42] pb-4 last:border-0">
            <a
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#569cd6] hover:text-[#6ab0de] transition-colors font-medium text-lg flex items-center group"
            >
              <LinkIcon
                className="mr-2 group-hover:translate-x-0.5 transition-transform"
                size={18}
              />
              {link.title}
            </a>
            {link.description && (
              <p className="text-[#9d9d9d] mt-2 pl-7 leading-relaxed">{link.description}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};
