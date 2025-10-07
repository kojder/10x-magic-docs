import { ReactNode } from "react";
import TopBar from "./TopBar";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-white dark:bg-[#0a0a0f] text-neutral-900 dark:text-neutral-100 flex flex-col transition-colors relative overflow-hidden">
      {/* Gradient background - visible in dark mode */}
      <div className="fixed inset-0 pointer-events-none opacity-0 dark:opacity-100 transition-opacity duration-500">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/4 right-1/4 w-[32rem] h-[32rem] bg-purple-600/20 rounded-full blur-3xl animate-pulse [animation-delay:1s]"></div>
        <div className="absolute bottom-1/4 left-1/3 w-80 h-80 bg-indigo-600/15 rounded-full blur-3xl animate-pulse [animation-delay:2s]"></div>
        <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-violet-600/15 rounded-full blur-3xl animate-pulse [animation-delay:1.5s]"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        <header className="bg-neutral-50/80 dark:bg-[#0f0f14]/80 backdrop-blur-xl border-b border-neutral-200 dark:border-neutral-800/50 sticky top-0 z-50 transition-colors">
          <div className="container max-w-7xl mx-auto p-4">
            <TopBar />
          </div>
        </header>
        <main className="flex-1 flex flex-col">
          <div className="container max-w-7xl mx-auto px-4 py-8 flex-1">{children}</div>
        </main>
        <footer className="bg-neutral-50/80 dark:bg-[#0f0f14]/80 backdrop-blur-xl border-t border-neutral-200 dark:border-neutral-800/50 py-6 transition-colors">
          <div className="container max-w-7xl mx-auto px-4 text-center text-neutral-600 dark:text-neutral-400">
            <p>© {new Date().getFullYear()} 10xDevs. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
