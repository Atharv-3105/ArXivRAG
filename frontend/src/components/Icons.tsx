// Stroke icons, drawn for this app (24px grid, currentColor).
type P = { size?: number };
const base = (size = 20) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export const ChatIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M4 5h16v11H9l-5 4z" />
  </svg>
);
export const ReviewIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <rect x="3" y="6" width="14" height="14" rx="2" />
    <path d="M7 3h14v14" />
  </svg>
);
export const ReadIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M3 5c3-1 6-1 9 1 3-2 6-2 9-1v14c-3-1-6-1-9 1-3-2-6-2-9-1z" />
    <path d="M12 6v14" />
  </svg>
);
export const MapIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <circle cx="6" cy="7" r="2.5" />
    <circle cx="18" cy="6" r="2.5" />
    <circle cx="12" cy="18" r="2.5" />
    <path d="M8 8l3 8M16 8l-3 8" />
  </svg>
);
export const CheckIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M5 12l4 4 10-10" />
  </svg>
);
export const HistoryIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M4 12a8 8 0 1 0 2.3-5.6L4 8.7" />
    <path d="M4 4v4.7h4.7M12 8v4l3 2" />
  </svg>
);
export const CloseIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);
export const PlusIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);
export const CopyIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V6a2 2 0 0 1 2-2h8" />
  </svg>
);
export const ThumbUpIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M7 11v9H4v-9zM7 11l4-7c1.5 0 2.5 1 2.2 2.6L12.6 9H19a2 2 0 0 1 2 2.4l-1.2 6A2 2 0 0 1 17.8 19H7" />
  </svg>
);
export const ThumbDownIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M7 13V4H4v9zM7 13l4 7c1.5 0 2.5-1 2.2-2.6L12.6 15H19a2 2 0 0 0 2-2.4l-1.2-6A2 2 0 0 0 17.8 5H7" />
  </svg>
);
export const StopIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);
export const ChevronIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);
export const ArrowDownIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M12 5v14M6 13l6 6 6-6" />
  </svg>
);
export const EyeIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
export const MoonIcon = ({ size }: P) => (
  <svg {...base(size)}>
    <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />
  </svg>
);
