import { useEffect } from "react";

const APP = "ArXivMind";

/** Keep the browser tab (and history entries) readable: "Reviews · ArXivMind". */
export function useTitle(title: string | null | undefined): void {
  useEffect(() => {
    document.title = title ? `${title.length > 60 ? `${title.slice(0, 57)}…` : title} · ${APP}` : APP;
    return () => {
      document.title = APP;
    };
  }, [title]);
}
