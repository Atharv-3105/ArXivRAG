import { Suspense, lazy, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router";
import { getToken, takeExpired } from "./api/client";
import { Rail } from "./components/Rail";
import { Chat } from "./pages/Chat";
import { Login } from "./pages/Login";
import { Me } from "./pages/Me";
import { Reading } from "./pages/Reading";
import { Reviews } from "./pages/Reviews";

// The map pulls in d3 and its layout: load it only when someone opens the map.
const MapPage = lazy(() => import("./pages/MapPage").then((m) => ({ default: m.MapPage })));

function useSignedIn(): boolean {
  const [signedIn, setSignedIn] = useState(() => Boolean(getToken()));
  useEffect(() => {
    const sync = () => setSignedIn(Boolean(getToken()));
    window.addEventListener("auth:change", sync);
    window.addEventListener("storage", sync); // another tab signed in or out
    return () => {
      window.removeEventListener("auth:change", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return signedIn;
}

function Shell({ signedIn }: { signedIn: boolean }) {
  if (!signedIn) return <Navigate to="/login" replace />;
  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Rail />
      <main id="main" className="shell-main">
        <Outlet />
      </main>
    </div>
  );
}

export function App() {
  const signedIn = useSignedIn();
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={signedIn ? <Navigate to={takeExpired() ?? "/"} replace /> : <Login />} />
        <Route element={<Shell signedIn={signedIn} />}>
          <Route index element={<Chat />} />
          <Route path="/chat/:conversationId" element={<Chat />} />
          <Route path="/reviews" element={<Reviews />} />
          <Route path="/reading" element={<Reading />} />
          <Route
            path="/map"
            element={
              <Suspense fallback={<p className="muted page-plain">Loading the map…</p>}>
                <MapPage />
              </Suspense>
            }
          />
          <Route path="/me" element={<Me />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
