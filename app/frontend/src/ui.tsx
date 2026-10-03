import { useEffect, useRef, useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Camera, ImageSquare, WarningCircle } from "@phosphor-icons/react";
import { api, type Sample, type Source } from "./api";

export const BLUE = "#5e97c4";
export const GRAPHITE = "#2a2c2f";

export const btnPrimary =
  "inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-[6px] bg-graphite px-5 font-medium text-sheet transition-[background-color,scale] duration-150 ease-(--ease-out) hover:bg-[#1d1f21] active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-45 disabled:active:scale-100";
export const btnSecondary =
  "inline-flex min-h-11 items-center justify-center gap-2 rounded-[6px] border border-graphite/80 px-5 font-medium text-graphite transition-[background-color,scale] duration-150 ease-(--ease-out) hover:bg-graphite/6 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-45 disabled:active:scale-100";

export function PageHeader({ title, line, aside }: { title: string; line: string; aside?: ReactNode }) {
  return (
    <header className="mb-10 flex flex-wrap items-end justify-between gap-4 lg:mt-2">
      <div>
        <h1 className="font-display text-[clamp(2.1rem,4vw,3.25rem)] leading-[1.02] font-bold tracking-[-0.03em] text-balance">
          {title}
        </h1>
        <p className="mt-3 max-w-[60ch] text-[17px] text-soft">{line}</p>
      </div>
      {aside}
    </header>
  );
}

/** Controls column + stage. One column under 768px. */
export function Workspace({ controls, stage }: { controls: ReactNode; stage: ReactNode }) {
  return (
    <div className="grid items-start gap-6 md:grid-cols-[288px_minmax(0,1fr)] xl:grid-cols-[320px_minmax(0,1fr)]">
      <div className="sheet divide-y divide-graphite/12 [&>*]:px-5 [&>*]:py-5">{controls}</div>
      <div className="min-w-0">{stage}</div>
    </div>
  );
}

export function Field({ legend, children, note }: { legend: string; children: ReactNode; note?: ReactNode }) {
  return (
    <fieldset className="min-w-0 [&>legend+*]:clear-both">
      <legend className="float-left mb-1 w-full text-sm font-medium text-soft">{legend}</legend>
      {children}
      {note && <p className="mt-2 text-sm text-soft">{note}</p>}
    </fieldset>
  );
}

/** Radio group drawn as text with a blue underline on the chosen option. Arrow keys work natively. */
export function Segmented<T extends string | number>({
  name,
  options,
  value,
  onChange,
  disabled,
}: {
  name: string;
  options: readonly (readonly [T, string])[];
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className={`flex flex-wrap gap-x-4 ${disabled ? "opacity-45" : ""}`}>
      {options.map(([v, label]) => (
        <label
          key={String(v)}
          className="relative flex min-h-11 cursor-pointer items-center text-soft has-checked:text-graphite has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-blue has-disabled:cursor-not-allowed"
        >
          <input
            type="radio"
            name={name}
            className="peer sr-only"
            checked={v === value}
            disabled={disabled}
            onChange={() => onChange(v)}
          />
          <span className="border-b-2 border-transparent py-0.5 peer-checked:border-blue">{label}</span>
        </label>
      ))}
    </div>
  );
}

export function ErrorLine({ children }: { children: ReactNode }) {
  if (!children) return null;
  return (
    <p role="alert" className="mt-3 flex gap-2 text-sm text-brick">
      <WarningCircle weight="light" size={18} className="mt-0.5 shrink-0" aria-hidden />
      <span>{children}</span>
    </p>
  );
}

/** A square model image with its caption. Skeleton while loading, empty frame before a run. */
export function Plate({
  src,
  caption,
  meta,
  loading,
  empty,
  className = "",
}: {
  src?: string;
  caption: ReactNode;
  meta?: ReactNode;
  loading?: boolean;
  empty?: string;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <figure className={`min-w-0 ${className}`}>
      <div className="registered">
        {loading ? (
          <div className="hatch aspect-square" aria-label="Working" />
        ) : src ? (
          // fade in from blurred greyscale
          <motion.img
            key={src}
            src={src}
            alt={typeof caption === "string" ? caption : ""}
            className="plate-img"
            initial={reduce ? false : { opacity: 0, filter: "blur(6px) grayscale(1)" }}
            animate={{ opacity: 1, filter: "blur(0px) grayscale(0)" }}
            transition={{ duration: 0.55, ease: [0.23, 1, 0.32, 1] }}
          />
        ) : (
          <div className="grid aspect-square place-items-center rounded-[4px] bg-graphite/[0.035] p-6 text-center text-sm text-soft">
            {empty}
          </div>
        )}
      </div>
      <figcaption className="mt-1.5 flex items-baseline justify-between gap-3 px-[7px] text-sm text-soft">
        <span className="text-graphite">{caption}</span>
        {meta && <span className="font-mono text-xs">{meta}</span>}
      </figcaption>
    </figure>
  );
}

/** Two plates, same size, same baseline. */
export function PlatePair({ children }: { children: ReactNode }) {
  return <div className="grid gap-6 md:grid-cols-2">{children}</div>;
}

/** A graphite stroke whose length is the value. No track; faint ticks at 0 and 1. Chosen strokes turn blue last. */
export function Stroke({ value, chosen, index = 0 }: { value: number; chosen?: boolean; index?: number }) {
  const reduce = useReducedMotion();
  return (
    <div className="relative h-4 min-w-0" aria-hidden>
      <span className="absolute top-1/2 left-0 h-2.5 w-px -translate-y-1/2 bg-smudge/70" />
      <span className="absolute top-1/2 right-0 h-2.5 w-px -translate-y-1/2 bg-smudge/70" />
      <motion.span
        className="absolute top-[calc(50%-1.5px)] left-0 h-[3px] origin-left rounded-full"
        style={{ width: `${Math.max(value, 0) * 100}%` }}
        initial={reduce ? false : { scaleX: 0, backgroundColor: GRAPHITE }}
        animate={{ scaleX: 1, backgroundColor: chosen ? BLUE : GRAPHITE }}
        transition={{
          scaleX: { type: "spring", stiffness: 100, damping: 20, delay: index * 0.06 },
          backgroundColor: { delay: 0.55, duration: 0.25 },
        }}
      />
    </div>
  );
}

export const fmt = (v: number, d = 3) => v.toFixed(d);

export function Readouts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="mt-6 flex flex-col gap-x-10 gap-y-3 border-t border-graphite/12 pt-4 sm:flex-row sm:flex-wrap">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-sm text-soft">{k}</dt>
          <dd className="font-mono text-sm text-graphite">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function EmptyStage() {
  return (
    <div className="sheet grid min-h-[22rem] place-items-center p-8 text-center">
      <div>
        <div className="registered mx-auto grid size-28 place-items-center">
          <ImageSquare weight="thin" size={44} className="text-soft" aria-hidden />
        </div>
        <p className="mt-5 font-display text-xl font-semibold tracking-[-0.01em]">Choose a sample or drop an image</p>
        <p className="mt-1 text-sm text-soft">Your result shows up here, next to the image it came from.</p>
      </div>
    </div>
  );
}

/** Generic request state for one workspace. */
export function useRun<T>() {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const run = async (fn: () => Promise<T>) => {
    setLoading(true);
    setError("");
    try {
      setData(await fn());
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };
  const reset = () => {
    setData(null);
    setError("");
  };
  return { data, error, loading, run, reset };
}

function useSamples(kind: "pets" | "faces") {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    api.samples(kind).then(setSamples, (e: Error) => setError(e.message));
  }, [kind]);
  return { samples, error };
}

const NOT_IMAGE = "This file is not an image. Choose a PNG or JPG.";

function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");
  const take = (f?: File | null) => {
    if (!f) return;
    if (!/^image\/(png|jpe?g)$/.test(f.type)) return setError(NOT_IMAGE);
    setError("");
    onFile(f);
  };
  return (
    <div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files[0]);
        }}
        className={`flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-[6px] border border-dashed px-4 py-5 text-center transition-colors has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-blue ${
          over ? "border-blue bg-blue/8" : error ? "border-brick" : "border-graphite/30"
        }`}
      >
        <input
          type="file"
          accept="image/png,image/jpeg"
          className="sr-only"
          onChange={(e) => {
            take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <span>Drop an image or choose a file</span>
        <span className="mt-1 text-sm text-soft">PNG or JPG, any size; resized to 128 x 128</span>
      </label>
      <ErrorLine>{error}</ErrorLine>
    </div>
  );
}

function SampleStrip({ kind, source, onPick }: { kind: "pets" | "faces"; source: Source | null; onPick: (s: Source) => void }) {
  const { samples, error } = useSamples(kind);
  if (error) return <ErrorLine>Could not load the samples. {error}</ErrorLine>;
  return (
    <div role="radiogroup" aria-label="Samples" className="grid grid-cols-6 gap-1.5">
      {samples.length === 0
        ? Array.from({ length: 6 }, (_, i) => <div key={i} className="hatch aspect-square" />)
        : samples.map((s) => {
            const on = source?.kind === "sample" && source.id === s.id;
            return (
              <button
                key={s.id}
                type="button"
                role="radio"
                aria-checked={on}
                aria-label={s.label}
                title={s.label}
                onClick={() => onPick({ kind: "sample", id: s.id, url: s.url })}
                className={`min-h-11 rounded-[4px] p-0.5 outline-offset-1 ${on ? "outline-2 outline-blue" : "hover:outline-1 hover:outline-graphite/40"}`}
              >
                <img src={s.url} alt="" className="plate-img" />
              </button>
            );
          })}
    </div>
  );
}

function cameraError(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError")
    return "Camera access was blocked. Allow the camera for this page in your browser's site settings, or upload a photo instead.";
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return "No camera was found. Connect one, or upload a photo instead.";
  if (name === "NotReadableError") return "The camera is in use by another app. Close it and try again.";
  return "The camera could not start. Upload a photo or pick a sample instead.";
}

function Webcam({ onCapture }: { onCapture: (f: File) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState("");
  const [live, setLive] = useState(false);
  useEffect(() => {
    let stream: MediaStream | undefined;
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("The camera only works on a secure page (localhost or https). Upload a photo instead.");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user", width: 640, height: 640 } })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        video.current!.srcObject = s;
        setLive(true);
      })
      .catch((e) => setError(cameraError(e)));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);
  const capture = () => {
    const v = video.current!;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    c.toBlob((b) => b && onCapture(new File([b], "webcam.png", { type: "image/png" })), "image/png");
  };
  if (error) return <ErrorLine>{error}</ErrorLine>;
  return (
    <div>
      <div className={live ? "" : "hatch aspect-square"}>
        <video ref={video} autoPlay playsInline muted className={`plate-img -scale-x-100 ${live ? "" : "invisible"}`} />
      </div>
      <button type="button" className={`${btnSecondary} mt-3 w-full`} onClick={capture} disabled={!live}>
        <Camera weight="light" size={20} aria-hidden /> Capture photo
      </button>
    </div>
  );
}

/** Source chooser shared by every workspace: Upload, Sample, and (faces only) Webcam. */
export function SourcePicker({
  kind,
  source,
  onChange,
  webcam,
}: {
  kind: "pets" | "faces";
  source: Source | null;
  onChange: (s: Source) => void;
  webcam?: boolean;
}) {
  type Tab = "upload" | "webcam" | "sample";
  const [tab, setTab] = useState<Tab>("sample");
  const tabs: [Tab, string][] = webcam
    ? [["upload", "Upload"], ["webcam", "Webcam"], ["sample", "Sample"]]
    : [["upload", "Upload"], ["sample", "Sample"]];
  const fromFile = (file: File) => onChange({ kind: "file", file, url: URL.createObjectURL(file) });
  return (
    <Field legend="Source">
      <Segmented name={`source-${kind}`} options={tabs} value={tab} onChange={setTab} />
      <div className="mt-3">
        {tab === "sample" && <SampleStrip kind={kind} source={source} onPick={onChange} />}
        {tab === "upload" && <DropZone onFile={fromFile} />}
        {tab === "webcam" && <Webcam onCapture={fromFile} />}
      </div>
      {source?.kind === "file" && tab !== "sample" && (
        <p className="mt-2 truncate text-sm text-soft">
          Using <span className="text-graphite">{source.file.name}</span>
        </p>
      )}
    </Field>
  );
}

/** One plate with a vertical blue line: input on the left, output on the right. Pointer or arrow keys. */
export function Wipe({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50);
  const box = useRef<HTMLDivElement>(null);
  const set = (v: number) => setPos(Math.min(100, Math.max(0, v)));
  const follow = (e: React.PointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    set(((e.clientX - r.left) / r.width) * 100);
  };
  return (
    <figure className="mx-auto w-full max-w-[min(100%,560px)]">
      <div className="registered">
      <div
        ref={box}
        className="relative cursor-ew-resize touch-none select-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          follow(e);
        }}
        onPointerMove={(e) => e.buttons && follow(e)}
      >
        <img src={before} alt="Input" className="plate-img" draggable={false} />
        <img
          src={after}
          alt="Restored"
          className="plate-img absolute inset-0"
          style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
          draggable={false}
        />
        <span className="absolute inset-y-0 w-px -translate-x-1/2 bg-blue" style={{ left: `${pos}%` }} />
        <span
          className="absolute -top-5 -translate-x-1/2 font-mono text-[11px] whitespace-nowrap text-blue"
          style={{ left: `${pos}%` }}
        >
          drag to compare
        </span>
        <span
          role="slider"
          tabIndex={0}
          aria-label="Wipe position"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pos)}
          aria-valuetext={`${Math.round(pos)}% input, ${100 - Math.round(pos)}% restored`}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10 : 2;
            if (e.key === "ArrowLeft" || e.key === "ArrowDown") set(pos - step);
            else if (e.key === "ArrowRight" || e.key === "ArrowUp") set(pos + step);
            else if (e.key === "Home") set(0);
            else if (e.key === "End") set(100);
            else return;
            e.preventDefault();
          }}
          className="absolute top-1/2 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full"
          style={{ left: `${pos}%` }}
        >
          <span className="size-4 rounded-full border-2 border-blue bg-sheet shadow-[0_1px_3px_rgb(42_44_47/0.25)]" />
        </span>
      </div>
      </div>
      <figcaption className="mt-2 flex justify-between text-sm">
        <span>Input</span>
        <span>Restored</span>
      </figcaption>
    </figure>
  );
}
