// Light, dark, or follow the system. The choice is a per-device convenience, so it lives in
// localStorage and the page works if that is unavailable.

export type Theme = "system" | "light" | "dark";
const KEY = "arxivmind.theme";

export function getTheme(): Theme {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

export function setTheme(theme: Theme): void {
  try {
    if (theme === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, theme);
  } catch {
    /* private mode: applies until the tab closes */
  }
  applyTheme(theme);
}
