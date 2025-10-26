import { ReactNode } from "react";
import TopBar from "./TopBar";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] flex flex-col transition-colors relative overflow-hidden">
      {/* Subtle gradient background for depth */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-[#0e639c]/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/4 right-1/4 w-[32rem] h-[32rem] bg-[#4ec9b0]/8 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 left-1/3 w-80 h-80 bg-[#569cd6]/8 rounded-full blur-3xl"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        <header className="bg-[#252526]/95 backdrop-blur-xl border-b border-[#3e3e42] sticky top-0 z-50 transition-colors shadow-lg">
          <div className="container max-w-7xl mx-auto p-4">
            <TopBar />
          </div>
        </header>
        <main className="flex-1 flex flex-col">
          <div className="container max-w-7xl mx-auto px-4 py-8 flex-1">{children}</div>
        </main>
        <footer className="bg-[#252526]/95 backdrop-blur-xl border-t border-[#3e3e42] py-6 transition-colors">
          <div className="container max-w-7xl mx-auto px-4 text-center text-[#858585]">
            <p>© {new Date().getFullYear()} 10xDevs. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
