import { TextBlock } from "../../tools/TextBlock";
import { CodeSnippet } from "../../tools/CodeSnippet";
import { MermaidDiagram } from "../../tools/MermaidDiagram";
import { Quiz } from "../../tools/Quiz";
import { Resources } from "../../tools/Resources";
import { QuizQuestion } from "../../tools/tools.types";

export default function GitHubActionsSonarQubePage() {
  const quiz1: QuizQuestion = {
    question: "What is the primary purpose of integrating SonarQube Cloud with GitHub Actions?",
    options: [
      { id: "A", text: "To automatically deploy applications to production" },
      { id: "B", text: "To perform automated code quality and security analysis" },
      { id: "C", text: "To manage GitHub repository permissions" },
      { id: "D", text: "To compile and build source code faster" },
    ],
    correctAnswer: "B",
    explanation:
      "SonarQube Cloud integration with GitHub Actions enables continuous inspection of code quality and security vulnerabilities as part of your CI/CD pipeline, providing automated static analysis on every commit or pull request.",
  };

  const quiz2: QuizQuestion = {
    question:
      "Which secret must be configured in GitHub repository settings for SonarQube Cloud integration?",
    options: [
      { id: "A", text: "GITHUB_TOKEN only" },
      { id: "B", text: "SONAR_TOKEN only" },
      { id: "C", text: "Both SONAR_TOKEN and SONAR_HOST_URL" },
      { id: "D", text: "API_KEY and PROJECT_ID" },
    ],
    correctAnswer: "C",
    explanation:
      "You need to configure both SONAR_TOKEN (authentication token from SonarQube Cloud) and SONAR_HOST_URL (typically https://sonarcloud.io) as GitHub secrets to enable secure communication between GitHub Actions and SonarQube Cloud.",
  };

  const quiz3: QuizQuestion = {
    question: "What happens when code quality fails to meet the defined Quality Gate in SonarQube?",
    options: [
      { id: "A", text: "The code is automatically refactored" },
      { id: "B", text: "The workflow fails and blocks the pull request merge" },
      { id: "C", text: "A warning email is sent to developers" },
      { id: "D", text: "The repository is locked temporarily" },
    ],
    correctAnswer: "B",
    explanation:
      "When code fails the Quality Gate, the GitHub Actions workflow returns a non-zero exit code, causing the check to fail. This prevents merging pull requests until quality standards are met, enforcing code quality policies at the CI/CD level.",
  };

  return (
    <div className="space-y-12">
      <h1 className="text-4xl font-bold mb-8 text-white">
        GitHub Actions + SonarQube Cloud: Automated Code Quality and Security Analysis
      </h1>

      <TextBlock
        header="The Problem: Code Quality Drift in Modern Development"
        text="In fast-paced development environments, maintaining consistent code quality and security standards becomes increasingly challenging. Manual code reviews are time-consuming, subjective, and often miss subtle security vulnerabilities or technical debt accumulation. Without automated quality gates, teams face:\n\n- **Inconsistent code standards** across team members and repositories\n- **Security vulnerabilities** that go undetected until production\n- **Technical debt accumulation** that compounds over time\n- **Delayed feedback loops** where issues are discovered late in the development cycle\n- **Merge conflicts and quality disputes** during pull request reviews\n\nGitHub Actions combined with SonarQube Cloud addresses these challenges by automating static code analysis, providing immediate feedback on code quality, security vulnerabilities, code coverage, and maintainability metrics directly within your CI/CD pipeline."
      />

      <MermaidDiagram
        diagramPath="/diagrams/github-actions-sonarqube-flow.mmd"
        caption="GitHub Actions workflow execution flow with SonarQube Cloud integration"
      />

      <TextBlock
        header="Conceptual Architecture: CI/CD Integration Pattern"
        text="The integration follows a **push-based continuous inspection** model where every code change triggers automated quality analysis:\n\n**1. Trigger Phase**\nGitHub Actions workflows are triggered by repository events (push, pull_request) and execute within GitHub-hosted or self-hosted runners.\n\n**2. Build and Test Phase**\nThe workflow checks out code, sets up the build environment (Node.js, Java, Python, etc.), and executes build scripts. Code coverage data is collected during test execution.\n\n**3. Analysis Phase**\nSonarScanner analyzes source code, applying configured quality profiles and rules. It identifies bugs, vulnerabilities, code smells, security hotspots, and calculates metrics like cyclomatic complexity and code duplication.\n\n**4. Quality Gate Evaluation**\nSonarQube Cloud evaluates the analysis results against predefined Quality Gate conditions (e.g., coverage > 80%, no critical vulnerabilities, maintainability rating A).\n\n**5. Reporting and Feedback**\nResults are published to SonarQube Cloud dashboard and reported back to GitHub as check runs, with inline comments on pull requests highlighting specific issues with file locations and remediation guidance."
      />

      <Quiz question={quiz1} />

      <TextBlock
        header="Implementation Architecture: Components and Flow"
        text="The implementation consists of several integrated components:\n\n**GitHub Actions Workflow**\nYAML-defined automation that orchestrates the entire quality analysis pipeline, managing dependencies, environment variables, and execution order.\n\n**SonarScanner**\nStatic analysis engine that parses source code, applies language-specific rules, and generates comprehensive quality metrics. Available as CLI tool or language-specific scanners (Maven, Gradle, npm).\n\n**SonarQube Cloud Platform**\nSaaS-based code quality platform that stores analysis results, manages Quality Gates, tracks metrics over time, and provides web-based dashboards for visualization.\n\n**GitHub Integration**\nBidirectional integration enabling automatic project detection, pull request decoration with quality metrics, and status checks that can block merges based on quality gate results.\n\n**Authentication Layer**\nSecure token-based authentication using GitHub Secrets to store sensitive credentials (SONAR_TOKEN, SONAR_HOST_URL) without exposing them in source code or logs."
      />

      <CodeSnippet
        language="yaml"
        fileName=".github/workflows/sonarqube.yml"
        showLineNumbers={true}
        code={`name: SonarQube Code Analysis

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  sonarqube:
    name: SonarQube Analysis
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Shallow clones disabled for better analysis
      
      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      
      - name: Cache SonarQube packages
        uses: actions/cache@v4
        with:
          path: ~/.sonar/cache
          key: \${{ runner.os }}-sonar
          restore-keys: \${{ runner.os }}-sonar
      
      - name: Cache Maven packages
        uses: actions/cache@v4
        with:
          path: ~/.m2
          key: \${{ runner.os }}-m2-\${{ hashFiles('**/pom.xml') }}
          restore-keys: \${{ runner.os }}-m2
      
      - name: Build and analyze
        env:
          GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: \${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: \${{ secrets.SONAR_HOST_URL }}
        run: |
          mvn clean verify sonar:sonar \\
            -Dsonar.projectKey=my-organization_my-project \\
            -Dsonar.organization=my-organization \\
            -Dsonar.host.url=\${SONAR_HOST_URL} \\
            -Dsonar.token=\${SONAR_TOKEN}
      
      - name: Check Quality Gate
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: \${{ secrets.SONAR_TOKEN }}
        with:
          scanMetadataReportFile: target/sonar/report-task.txt`}
      />

      <Quiz question={quiz2} />

      <CodeSnippet
        language="yaml"
        fileName=".github/workflows/sonarqube-typescript.yml"
        showLineNumbers={true}
        code={`name: SonarQube TypeScript Analysis

on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  sonarqube:
    name: SonarQube TypeScript Analysis
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests with coverage
        run: npm run test:coverage
      
      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: \${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: \${{ secrets.SONAR_HOST_URL }}
        with:
          args: >
            -Dsonar.projectKey=my-typescript-project
            -Dsonar.organization=my-org
            -Dsonar.sources=src
            -Dsonar.tests=src
            -Dsonar.test.inclusions=**/*.test.ts,**/*.test.tsx
            -Dsonar.typescript.lcov.reportPaths=coverage/lcov.info
            -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
      
      - name: SonarQube Quality Gate check
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: \${{ secrets.SONAR_TOKEN }}`}
      />

      <TextBlock
        header="Configuration and Setup: Step-by-Step Implementation"
        text="**Step 1: SonarQube Cloud Setup**\nNavigate to [sonarcloud.io](https://sonarcloud.io) and authenticate with your GitHub account. Create a new organization (linked to your GitHub organization) and import your repository. SonarQube Cloud will automatically detect the project and suggest appropriate analysis configuration.\n\n**Step 2: Generate Authentication Token**\nIn SonarQube Cloud, navigate to Account Settings > Security > Generate Tokens. Create a token with appropriate permissions for your project. Copy this token immediately as it won't be shown again.\n\n**Step 3: Configure GitHub Secrets**\nIn your GitHub repository, go to Settings > Secrets and variables > Actions. Add two repository secrets:\n- `SONAR_TOKEN`: Your generated SonarQube token\n- `SONAR_HOST_URL`: `https://sonarcloud.io`\n\n**Step 4: Create sonar-project.properties**\nFor Maven/Gradle projects, configuration can be in pom.xml/build.gradle. For other projects, create a `sonar-project.properties` file in the repository root with project-specific settings.\n\n**Step 5: Configure Quality Gates**\nIn SonarQube Cloud project settings, customize Quality Gate conditions to match your team's standards. Common conditions include: code coverage percentage, duplicated lines density, reliability rating, security rating, and maintainability rating.\n\n**Step 6: Enable Pull Request Decoration**\nIn SonarQube Cloud, navigate to Administration > General Settings > Pull Requests. Ensure GitHub integration is enabled to display quality metrics directly on pull requests.\n\n**Step 7: Test the Integration**\nCreate a pull request with intentional code quality issues to verify that SonarQube analysis runs, quality gates are evaluated, and results appear in both GitHub and SonarQube Cloud interfaces."
      />

      <CodeSnippet
        language="properties"
        fileName="sonar-project.properties"
        showLineNumbers={true}
        code={`# Project identification
sonar.projectKey=my-organization_my-project
sonar.organization=my-organization

# Project metadata
sonar.projectName=My Project Name
sonar.projectVersion=1.0.0

# Source code location
sonar.sources=src
sonar.tests=src
sonar.test.inclusions=**/*.test.ts,**/*.test.tsx,**/*.spec.ts

# Exclusions
sonar.exclusions=**/node_modules/**,**/dist/**,**/build/**,**/*.config.ts
sonar.test.exclusions=**/node_modules/**,**/dist/**

# Coverage reports
sonar.javascript.lcov.reportPaths=coverage/lcov.info
sonar.typescript.lcov.reportPaths=coverage/lcov.info

# Language-specific settings
sonar.sourceEncoding=UTF-8
sonar.typescript.tsconfigPath=tsconfig.json

# Quality Gate settings
sonar.qualitygate.wait=true
sonar.qualitygate.timeout=300`}
      />

      <Quiz question={quiz3} />

      <CodeSnippet
        language="xml"
        fileName="pom.xml"
        showLineNumbers={true}
        code={`<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>my-spring-boot-app</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <java.version>17</java.version>
        <spring-boot.version>3.2.0</spring-boot.version>
        <sonar.organization>my-organization</sonar.organization>
        <sonar.projectKey>my-organization_my-project</sonar.projectKey>
        <sonar.host.url>https://sonarcloud.io</sonar.host.url>
        <sonar.coverage.jacoco.xmlReportPaths>
            \${project.build.directory}/site/jacoco/jacoco.xml
        </sonar.coverage.jacoco.xmlReportPaths>
    </properties>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.11</version>
                <executions>
                    <execution>
                        <goals>
                            <goal>prepare-agent</goal>
                        </goals>
                    </execution>
                    <execution>
                        <id>report</id>
                        <phase>test</phase>
                        <goals>
                            <goal>report</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
            
            <plugin>
                <groupId>org.sonarsource.scanner.maven</groupId>
                <artifactId>sonar-maven-plugin</artifactId>
                <version>3.10.0.2594</version>
            </plugin>
        </plugins>
    </build>
</project>`}
      />

      <TextBlock
        header="Advanced Patterns and Best Practices"
        text="**Monorepo Analysis Strategy**\nFor monorepos, configure multiple SonarQube projects with separate project keys or use a single project with modular analysis. Leverage `sonar.sources` and exclusion patterns to analyze specific subdirectories independently.\n\n**Branch Analysis and PR Decoration**\nEnable branch analysis to track quality metrics across feature branches. SonarQube Cloud provides differential analysis showing only new code issues introduced in pull requests, preventing historical debt from blocking new features.\n\n**Performance Optimization**\nImplement caching strategies for SonarScanner and dependency managers (Maven, npm) to reduce workflow execution time. Use `fetch-depth: 0` only when blame information is required; otherwise shallow clones are sufficient.\n\n**Quality Gate Customization**\nCreate organization-level Quality Gate templates that can be applied across multiple projects. Consider separate gates for legacy codebases (focusing on new code) versus greenfield projects (applying stricter overall metrics).\n\n**Security-First Configuration**\nEnable Security Hotspots detection and configure OWASP Top 10 rules. Integrate SonarQube results with GitHub Security tab using SARIF export for centralized vulnerability management.\n\n**Multi-Language Projects**\nSonarQube Cloud supports polyglot analysis. Configure language-specific scanners and quality profiles while maintaining unified quality gates. Ensure test coverage reports are correctly mapped for each language ecosystem.\n\n**Workflow Optimization Patterns**\nImplement conditional workflow execution using path filters to avoid unnecessary analyses when non-code files change. Use matrix strategies for multi-module or multi-language projects requiring different build configurations."
      />

      <Resources
        title="Essential Resources and Documentation"
        links={[
          {
            title: "SonarQube Cloud Official Documentation",
            url: "https://docs.sonarcloud.io/",
            description:
              "Comprehensive documentation covering setup, configuration, and advanced features of SonarQube Cloud platform",
          },
          {
            title: "GitHub Actions Documentation",
            url: "https://docs.github.com/en/actions",
            description:
              "Complete guide to GitHub Actions workflows, syntax, and best practices for CI/CD automation",
          },
          {
            title: "SonarScanner for Maven",
            url: "https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-maven/",
            description:
              "Integration guide for analyzing Java projects using Maven build tool with SonarQube",
          },
          {
            title: "SonarScanner for JavaScript/TypeScript",
            url: "https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner/",
            description:
              "Documentation for analyzing JavaScript and TypeScript codebases using SonarScanner CLI",
          },
          {
            title: "Quality Gates Configuration Guide",
            url: "https://docs.sonarcloud.io/improving/quality-gates/",
            description:
              "Learn how to define and customize quality gates to enforce code quality standards in your organization",
          },
          {
            title: "GitHub-SonarQube Integration",
            url: "https://docs.sonarcloud.io/getting-started/github/",
            description:
              "Step-by-step guide for integrating SonarQube Cloud with GitHub repositories and pull requests",
          },
          {
            title: "SonarQube Rules Explorer",
            url: "https://rules.sonarsource.com/",
            description:
              "Browse and search all available code analysis rules across different programming languages",
          },
          {
            title: "Code Coverage Best Practices",
            url: "https://docs.sonarcloud.io/improving/test-coverage/overview/",
            description:
              "Understanding code coverage metrics and strategies for improving test coverage in your projects",
          },
        ]}
      />
    </div>
  );
}
