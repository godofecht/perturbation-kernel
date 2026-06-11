"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { Preferences } from "./types";

// Minimal client store: the user's matching preferences plus which meals they
// have joined (and the buddies they invited). Persisted to localStorage so the
// clickable demo survives a refresh. No backend required.

export interface JoinedTable {
  mealId: string;
  buddyIds: string[];
}

interface StoreState {
  prefs: Preferences;
  setPrefs: (p: Preferences) => void;
  joined: JoinedTable[];
  isJoined: (mealId: string) => boolean;
  joinTable: (mealId: string, buddyIds: string[]) => void;
  leaveTable: (mealId: string) => void;
}

export const DEFAULT_PREFS: Preferences = {
  cuisines: [],
  dietary: [],
  neighborhood: "Brixton",
  sitting: "any",
  budget: 25,
  wantsBuddies: true,
};

const StoreContext = createContext<StoreState | null>(null);

const STORAGE_KEY = "meal-match:v1";

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [prefs, setPrefsState] = useState<Preferences>(DEFAULT_PREFS);
  const [joined, setJoined] = useState<JoinedTable[]>([]);
  const [hydrated, setHydrated] = useState(false);

  // Load once on mount.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as {
          prefs?: Preferences;
          joined?: JoinedTable[];
        };
        if (parsed.prefs) setPrefsState({ ...DEFAULT_PREFS, ...parsed.prefs });
        if (parsed.joined) setJoined(parsed.joined);
      }
    } catch {
      // ignore corrupt storage
    }
    setHydrated(true);
  }, []);

  // Persist on change (after first hydration).
  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ prefs, joined }));
  }, [prefs, joined, hydrated]);

  const setPrefs = useCallback((p: Preferences) => setPrefsState(p), []);

  const isJoined = useCallback(
    (mealId: string) => joined.some((j) => j.mealId === mealId),
    [joined],
  );

  const joinTable = useCallback((mealId: string, buddyIds: string[]) => {
    setJoined((prev) => {
      const others = prev.filter((j) => j.mealId !== mealId);
      return [...others, { mealId, buddyIds }];
    });
  }, []);

  const leaveTable = useCallback((mealId: string) => {
    setJoined((prev) => prev.filter((j) => j.mealId !== mealId));
  }, []);

  const value = useMemo<StoreState>(
    () => ({ prefs, setPrefs, joined, isJoined, joinTable, leaveTable }),
    [prefs, setPrefs, joined, isJoined, joinTable, leaveTable],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): StoreState {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used within a StoreProvider");
  return ctx;
}
