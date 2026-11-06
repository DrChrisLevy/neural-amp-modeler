import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"
type Theme = "dark" | "light"
const Context = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "dark",
  toggle: () => {},
})
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem("nam-theme") === "light" ? "light" : "dark"
    } catch {
      return "dark"
    }
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    try {
      localStorage.setItem("nam-theme", theme)
    } catch {
      /* Theme persistence is optional. */
    }
  }, [theme])
  return (
    <Context.Provider
      value={{
        theme,
        toggle: () =>
          setTheme((value) => (value === "dark" ? "light" : "dark")),
      }}
    >
      {children}
    </Context.Provider>
  )
}
export const useTheme = () => useContext(Context)
