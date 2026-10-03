// Client for the Loupe backend. With VITE_MOCK=1 every call is answered by ./mock instead.

export type CorruptionType = "none" | "salt" | "blur" | "occlusion";
export type Severity = "low" | "medium" | "high";
export type Expert = "identity" | "salt" | "blur" | "occlusion";

export interface Health {
  status: "ok" | "degraded";
  models: Record<string, boolean>;
}
export interface Sample {
  id: string;
  url: string;
  label: string;
}
export interface Restored {
  input_png: string;
  output_png: string;
  corruption: { type: CorruptionType; severity: Severity; label: string };
}
export interface UniversalResult extends Restored {
  inference_ms: number;
}
export interface HardResult extends Restored {
  probs: Record<"clean" | "salt" | "blur" | "occlusion", number>;
  predicted: "clean" | "salt" | "blur" | "occlusion";
  expert: Expert;
  inference_ms: { classifier: number; expert: number; total: number };
}
export interface SoftResult extends Restored {
  weights: Record<Expert, number>;
  dominant: Expert;
  branch_pngs: Record<Expert, string>;
  inference_ms: number;
}
export interface SketchResult {
  photo_png: string;
  sketch_png: string;
  style: 1 | 2 | 3;
  inference_ms: number;
  all_styles?: { style: 1 | 2 | 3; sketch_png: string }[];
}
export interface ExperimentGroup {
  task: string;
  facts: [string, string][];
}

/** What the user picked as input: a bundled sample or their own file. */
export type Source = { kind: "sample"; id: string; url: string } | { kind: "file"; file: File; url: string };

const MOCK = import.meta.env.VITE_MOCK === "1";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  if (MOCK) {
    const { mockFetch } = await import("./mock");
    return (await mockFetch(path, init)) as T;
  }
  let res: Response;
  try {
    res = await fetch("/api" + path, init);
  } catch {
    throw new Error("Could not reach the backend. Start it with docker compose up, then try again.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string" ? body.detail : `The backend answered with error ${res.status}. Try again.`,
    );
  }
  return res.json();
}

function form(source: Source, fields: Record<string, string>) {
  const f = new FormData();
  if (source.kind === "sample") f.set("sample_id", source.id);
  else f.set("image", source.file);
  for (const [k, v] of Object.entries(fields)) f.set(k, v);
  return { method: "POST", body: f };
}

export const api = {
  health: () => call<Health>("/health"),
  samples: (kind: "pets" | "faces") => call<Sample[]>(`/samples?kind=${kind}`),
  universal: (s: Source, corruption: CorruptionType, severity: Severity) =>
    call<UniversalResult>("/restore/universal", form(s, { corruption, severity })),
  hard: (s: Source, corruption: CorruptionType, severity: Severity) =>
    call<HardResult>("/restore/hard", form(s, { corruption, severity })),
  soft: (s: Source, corruption: CorruptionType, severity: Severity) =>
    call<SoftResult>("/restore/soft", form(s, { corruption, severity })),
  sketch: (s: Source, style: number, allStyles: boolean) =>
    call<SketchResult>("/sketch", form(s, { style: String(style), all_styles: String(allStyles) })),
  experiments: () => call<ExperimentGroup[]>("/experiments"),
  experimentsMeta: () => call<{ mlflow_url: string }>("/experiments/meta"),
};
