import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import "./lib/engines/comparison-audio.js"
import "./lib/engines/training-metrics.js"
import "./lib/engines/uploads.js"
import App from "./App"
import { ThemeProvider } from "@/components/theme-provider"
import { TooltipProvider } from "@/components/ui/tooltip"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <TooltipProvider delayDuration={300}>
        <App />
      </TooltipProvider>
    </ThemeProvider>
  </StrictMode>
)
