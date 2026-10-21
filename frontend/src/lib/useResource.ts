// Fetch-on-mount with the three states every screen needs: loading, failed (with a retry) and
// loaded. Race-safe: a response that arrives after the inputs changed, or after unmount, is
// dropped instead of overwriting newer data.

import { useCallback, useEffect, useState } from "react";
import { describeError } from "../api/client";

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

export function useResource<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[] = [],
): Resource<T> {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setState((s) => ({ data: s.data, error: null, loading: true }));
    load(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ data, error: null, loading: false });
      },
      (error) => {
        if (!controller.signal.aborted) setState({ data: null, error: describeError(error), loading: false });
      },
    );
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);
  return { ...state, reload };
}
