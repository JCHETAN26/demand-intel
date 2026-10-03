"use client";
import { useEffect, useState } from "react";
import { get } from "@/lib/api";

export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!path) return;
    let live = true;
    setError(null);
    get<T>(path).then((d) => live && setData(d)).catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, [path]);
  return { data, error };
}
