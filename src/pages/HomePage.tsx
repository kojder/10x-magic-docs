import { Link } from "react-router-dom";
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../components/ui/hover-card";

interface Subject {
  id: string;
  name: string;
  path: string;
  description: string;
}

// Introduce a new Subject to present it on the home page grid
const availableSubjects: Subject[] = [
  {
    id: "pytorch",
    name: "PyTorch Deep Learning",
    path: "/docs/pytorch",
    description:
      "Master deep learning fundamentals with PyTorch framework for research and production",
  },
  {
    id: "github-actions-sonarqube",
    name: "GitHub Actions + SonarQube",
    path: "/docs/github-actions-sonarqube",
    description:
      "Automated code quality and security analysis with GitHub Actions and SonarQube Cloud",
  },
];

// Total number of tiles to display
const TOTAL_TILES = 25;

export default function HomePage() {
  return (
    <div className="container mx-auto py-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        {Array.from({ length: TOTAL_TILES }, (_, index) => {
          const subject = availableSubjects[index];

          if (subject) {
            return (
              <HoverCard key={subject.id}>
                <HoverCardTrigger asChild>
                  <Link
                    to={subject.path}
                    className="block transition-all duration-200 hover:scale-105"
                  >
                    <Card
                      className="h-full border-2 border-[#3e3e42] shadow-lg bg-[#252526] backdrop-blur-sm hover:border-[#569cd6] transition-colors"
                      data-testid="card"
                    >
                      <CardHeader className="space-y-2">
                        <CardTitle className="text-xl font-bold">
                          <span className="text-[#d4d4d4]">{subject.name}</span>
                        </CardTitle>
                        <CardDescription className="text-sm text-[#9d9d9d] leading-relaxed">
                          {subject.description}
                        </CardDescription>
                      </CardHeader>
                      <CardFooter>
                        <Button className="w-full bg-[#0e639c] hover:bg-[#1177bb] text-white transition-all duration-200">
                          Start Learning
                        </Button>
                      </CardFooter>
                    </Card>
                  </Link>
                </HoverCardTrigger>
                <HoverCardContent className="w-80 bg-[#252526] border-[#3e3e42]">
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-[#d4d4d4]">{subject.name}</h4>
                    <p className="text-sm text-[#9d9d9d]">{subject.description}</p>
                  </div>
                </HoverCardContent>
              </HoverCard>
            );
          } else {
            return (
              <Card
                key={`placeholder-${index}`}
                className="h-full border border-dashed border-[#3e3e42] bg-[#252526]/50 backdrop-blur-sm opacity-60 transition-all duration-200 hover:opacity-100"
                data-testid="card"
              >
                <CardHeader className="space-y-2">
                  <CardTitle className="text-xl font-bold text-[#858585]">-</CardTitle>
                  <CardDescription className="text-[#858585]">
                    Use Agentic AI to create new learning paths
                  </CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button
                    disabled
                    className="w-full bg-[#3e3e42] text-[#858585]"
                    variant="secondary"
                  >
                    Subject {index + 1}
                  </Button>
                </CardFooter>
              </Card>
            );
          }
        })}
      </div>
    </div>
  );
}
