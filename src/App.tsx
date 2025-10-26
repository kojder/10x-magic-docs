import { RouterProvider, createBrowserRouter } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import PyTorchPage from "./components/docs/pytorch/PyTorchPage";
import GitHubActionsSonarQubePage from "./components/docs/github-actions-sonarqube/GitHubActionsSonarQubePage";

/* Each page should be wrapped in the Layout component */
const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Layout>
        <HomePage />
      </Layout>
    ),
  },
  {
    path: "/docs/pytorch",
    element: (
      <Layout>
        <PyTorchPage />
      </Layout>
    ),
  },
  {
    path: "/docs/github-actions-sonarqube",
    element: (
      <Layout>
        <GitHubActionsSonarQubePage />
      </Layout>
    ),
  },
]);

function App() {
  return (
    <div className="min-h-screen bg-[#1e1e1e]">
      <RouterProvider router={router} />
    </div>
  );
}

export default App;
