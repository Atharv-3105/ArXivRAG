import { percent } from "../lib/labels";

export function MasteryBar({ value, label }: { value: number; label: string }) {
  return (
    <div
      className="mastery-bar"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}
      aria-valuetext={percent(value)}
    >
      <span style={{ width: percent(value) }} />
    </div>
  );
}
