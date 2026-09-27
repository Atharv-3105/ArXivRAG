import { useEffect, useState } from "react";
import { NavLink } from "react-router";
import { api } from "../api/client";
import type { Me, MasteryRow } from "../api/types";
import { ChatIcon, MapIcon, ReadIcon, ReviewIcon } from "./Icons";

const items = [
  { to: "/", label: "Ask", icon: ChatIcon, end: true },
  { to: "/reviews", label: "Reviews", icon: ReviewIcon },
  { to: "/reading", label: "Read a paper", icon: ReadIcon },
  { to: "/map", label: "Paper map", icon: MapIcon },
];

export function Rail() {
  const [due, setDue] = useState(0);
  const [initial, setInitial] = useState("");

  useEffect(() => {
    api<MasteryRow[]>("/api/v1/me/reviews").then((rows) => setDue(rows.length), () => {});
    api<Me>("/api/v1/me").then(
      (me) => setInitial((me.display_name || me.email).charAt(0).toUpperCase()),
      () => {},
    );
  }, []);

  return (
    <nav className="rail" aria-label="Main">
      <NavLink to="/" className="rail-mark" aria-label="ArXivMind home">
        a
      </NavLink>
      {items.map(({ to, label, icon: Icon, end }) => (
        <NavLink key={to} to={to} end={end} className="rail-item" aria-label={label} title={label}>
          <Icon />
          {to === "/reviews" && due > 0 && (
            <span className="rail-badge" aria-label={`${due} due`}>
              {due}
            </span>
          )}
        </NavLink>
      ))}
      <span className="rail-spacer" />
      <NavLink to="/me" className="rail-avatar" aria-label="You: profile and progress" title="You">
        {initial || "·"}
      </NavLink>
    </nav>
  );
}
