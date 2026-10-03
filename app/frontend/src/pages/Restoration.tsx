import { useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { GitFork, MagicWand, Stack } from "@phosphor-icons/react";
import {
  api,
  type CorruptionType,
  type Expert,
  type HardResult,
  type Restored,
  type Severity,
  type SoftResult,
  type Source,
  type UniversalResult,
} from "../api";
import {
  btnPrimary,
  EmptyStage,
  ErrorLine,
  Field,
  fmt,
  PageHeader,
  Plate,
  PlatePair,
  Readouts,
  Segmented,
  SourcePicker,
  Stroke,
  useRun,
  Wipe,
  Workspace,
} from "../ui";

const CORRUPTIONS = [
  ["none", "None"],
  ["salt", "Salt-and-pepper"],
  ["blur", "Gaussian blur"],
  ["occlusion", "Occlusion"],
] as const;
const SEVERITIES = [
  ["low", "Low"],
  ["medium", "Medium"],
  ["high", "High"],
] as const;

type StageProps<T> = { r: T | null; loading: boolean; source: Source };

/** Shared controls column for the three restoration workspaces; each page draws its own stage. */
function RestorationWorkspace<T extends Restored>({
  title,
  line,
  button,
  icon,
  call,
  stage,
}: {
  title: string;
  line: string;
  button: string;
  icon: ReactNode;
  call: (s: Source, c: CorruptionType, v: Severity) => Promise<T>;
  stage: (p: StageProps<T>) => ReactNode;
}) {
  const [source, setSource] = useState<Source | null>(null);
  const [corruption, setCorruption] = useState<CorruptionType>("salt");
  const [severity, setSeverity] = useState<Severity>("medium");
  const { data, error, loading, run, reset } = useRun<T>();
  const changed =
    <V,>(set: (v: V) => void) =>
    (v: V) => {
      set(v);
      reset();
    };

  return (
    <>
      <PageHeader title={title} line={line} />
      <Workspace
        controls={
          <>
            <SourcePicker kind="pets" source={source} onChange={changed(setSource)} />
            <Field legend="Corruption">
              <Segmented name="corruption" options={CORRUPTIONS} value={corruption} onChange={changed(setCorruption)} />
            </Field>
            <Field
              legend="Severity"
              note={corruption === "none" ? "No corruption is added. Use this for an image that is already damaged." : undefined}
            >
              <Segmented
                name="severity"
                options={SEVERITIES}
                value={severity}
                onChange={changed(setSeverity)}
                disabled={corruption === "none"}
              />
            </Field>
            <div>
              <button
                type="button"
                className={btnPrimary}
                disabled={!source || loading}
                onClick={() => source && run(() => call(source, corruption, severity))}
              >
                {icon}
                {loading ? "Working" : button}
              </button>
              {!source && <p className="mt-2 text-sm text-soft">Pick a sample or upload an image first.</p>}
              <ErrorLine>{error}</ErrorLine>
            </div>
          </>
        }
        stage={
          source ? (
            <section aria-label="Result" aria-busy={loading} className="sheet p-5 md:p-6">
              {stage({ r: data, loading, source })}
            </section>
          ) : (
            <EmptyStage />
          )
        }
      />
    </>
  );
}

/** Input plate: the chosen image before a run, the corrupted model input after. */
function InputPlate({ r, loading, source }: StageProps<Restored>) {
  return (
    <Plate
      src={r?.input_png ?? source.url}
      caption="Input"
      meta={r ? "128 x 128" : "before corruption"}
      loading={loading}
    />
  );
}

const ms = (v: number) => `${fmt(v, 1)} ms`;

// ---------------------------------------------------------------- Universal

function UniversalStage(p: StageProps<UniversalResult>) {
  const [mode, setMode] = useState<"side" | "wipe">("side");
  const { r, loading } = p;
  return (
    <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-graphite/12">
        <Segmented
          name="compare"
          options={[
            ["side", "Side by side"],
            ["wipe", "Wipe"],
          ]}
          value={mode}
          onChange={setMode}
          disabled={!r}
        />
      </div>
      {mode === "wipe" && r && !loading ? (
        <div className="pt-6">
          <Wipe before={r.input_png} after={r.output_png} />
        </div>
      ) : (
        <PlatePair>
          <InputPlate {...p} />
          <Plate
            src={r?.output_png}
            caption="Restored"
            meta={r && "128 x 128"}
            loading={loading}
            empty="Press Restore to see the restored image."
          />
        </PlatePair>
      )}
      {r && (
        <Readouts
          items={[
            ["Corruption", r.corruption.label],
            ["Inference", ms(r.inference_ms)],
          ]}
        />
      )}
    </>
  );
}

export function Universal() {
  return (
    <RestorationWorkspace<UniversalResult>
      title="Universal Restoration"
      line="One autoencoder trained on every corruption type."
      button="Restore"
      icon={<MagicWand weight="light" size={20} aria-hidden />}
      call={api.universal}
      stage={(p) => <UniversalStage {...p} />}
    />
  );
}

// ---------------------------------------------------------------- Hard routing

const ROUTES = [
  { cls: "clean", label: "Clean", expert: "Identity" },
  { cls: "salt", label: "Salt-and-pepper", expert: "Salt specialist" },
  { cls: "blur", label: "Gaussian blur", expert: "Blur specialist" },
  { cls: "occlusion", label: "Occlusion", expert: "Occlusion specialist" },
] as const;

function Slot({ name, chosen, bypass }: { name: string; chosen: boolean; bypass: boolean }) {
  return (
    <div
      className={`flex min-h-10 items-center justify-between gap-2 rounded-[6px] border px-3 text-sm whitespace-nowrap transition-colors duration-300 ${
        chosen ? "border-blue text-graphite" : "border-graphite/15 text-soft"
      }`}
    >
      <span>{chosen && bypass ? "Identity bypass" : name}</span>
      {chosen && <span className="font-mono text-[11px] text-blue max-sm:sr-only">chosen</span>}
    </div>
  );
}

function HardStage(p: StageProps<HardResult>) {
  const { r, loading } = p;
  const reduce = useReducedMotion();
  const bypass = r?.expert === "identity";
  return (
    <>
      <PlatePair>
        <InputPlate {...p} />
        <Plate
          src={r?.output_png}
          caption={bypass ? "Returned unchanged" : "Restored"}
          meta={r && "128 x 128"}
          loading={loading}
          empty="Press Route and restore to see the classifier's choice."
        />
      </PlatePair>

      {(r || loading) && (
        <div className="mt-8">
          <h2 className="mb-3 font-medium">Classifier probabilities and expert</h2>
          {loading || !r ? (
            <div className="hatch h-44" aria-label="Working" />
          ) : (
            <>
              <ul className="space-y-1">
                {ROUTES.map(({ cls, label, expert }, i) => {
                  const v = r.probs[cls];
                  const win = r.predicted === cls;
                  return (
                    <li
                      key={cls}
                      className="grid min-h-11 grid-cols-[8.25rem_3.25rem_minmax(3rem,1fr)] items-center gap-x-3 sm:grid-cols-[8.5rem_3.5rem_minmax(4rem,1fr)_13.5rem]"
                    >
                      <span className={win ? "text-graphite" : "text-soft"}>{label}</span>
                      <span className="text-right font-mono text-sm">{fmt(v)}</span>
                      <div className="relative">
                        <Stroke value={v} chosen={win} index={i} />
                        {win && (
                          // the construction line from the winning stroke to its expert
                          <motion.span
                            aria-hidden
                            className="absolute top-1/2 hidden h-px origin-left bg-blue sm:block"
                            style={{ left: `${v * 100}%`, right: "-2.75rem" }}
                            initial={reduce ? false : { scaleX: 0 }}
                            animate={{ scaleX: 1 }}
                            transition={{ delay: 0.6, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                          />
                        )}
                      </div>
                      <div className="hidden sm:ml-8 sm:block">
                        <Slot name={expert} chosen={win} bypass={bypass} />
                      </div>
                    </li>
                  );
                })}
              </ul>
              <div className="mt-4 grid grid-cols-2 gap-2 sm:hidden">
                {ROUTES.map(({ cls, expert }) => (
                  <Slot key={cls} name={expert} chosen={r.predicted === cls} bypass={bypass} />
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {r && (
        <Readouts
          items={[
            ["Corruption", r.corruption.label],
            ["Predicted", ROUTES.find((x) => x.cls === r.predicted)!.label],
            ["Expert", bypass ? "Identity bypass, image returned unchanged" : ROUTES.find((x) => x.cls === r.expert)!.expert],
            [
              "Inference",
              `${fmt(r.inference_ms.classifier, 1)} classifier + ${fmt(r.inference_ms.expert, 1)} expert = ${ms(r.inference_ms.total)}`,
            ],
          ]}
        />
      )}
    </>
  );
}

export function HardRouted() {
  return (
    <RestorationWorkspace<HardResult>
      title="Hard-Routed Restoration"
      line="A classifier names the corruption, and only that specialist restores the image."
      button="Route and restore"
      icon={<GitFork weight="light" size={20} aria-hidden />}
      call={api.hard}
      stage={(p) => <HardStage {...p} />}
    />
  );
}

// ---------------------------------------------------------------- Soft mixture of experts

const BRANCHES: { key: Expert; label: string }[] = [
  { key: "identity", label: "Identity" },
  { key: "salt", label: "Salt expert" },
  { key: "blur", label: "Blur expert" },
  { key: "occlusion", label: "Occlusion expert" },
];

function SoftStage(p: StageProps<SoftResult>) {
  const { r, loading } = p;
  const reduce = useReducedMotion();
  return (
    <>
      <PlatePair>
        <InputPlate {...p} />
        <Plate
          src={r?.output_png}
          caption="Blended output"
          meta={r && "128 x 128"}
          loading={loading}
          empty="Press Blend and restore to see the blended image."
        />
      </PlatePair>

      {(r || loading) && (
        <div className="mt-8">
          <h2 className="font-medium">Branch contributions</h2>
          <p className="mb-4 text-sm text-soft">Each branch restores the image on its own. Thumbnail opacity follows its weight.</p>
          <div className="grid grid-cols-2 gap-x-5 gap-y-6 sm:grid-cols-4">
            {BRANCHES.map(({ key, label }, i) => {
              const w = r?.weights[key] ?? 0;
              const top = r?.dominant === key;
              return (
                <figure key={key} className="min-w-0">
                  {loading || !r ? (
                    <div className="hatch aspect-square" />
                  ) : (
                    <div className="rounded-[4px] bg-sheet">
                      <motion.img
                        src={r.branch_pngs[key]}
                        alt={`${label} branch output`}
                        className="plate-img"
                        initial={reduce ? false : { opacity: 0 }}
                        animate={{ opacity: 0.12 + 0.88 * w }}
                        transition={{ duration: 0.5, delay: i * 0.06 }}
                      />
                    </div>
                  )}
                  <figcaption className="mt-2">
                    <div className="flex items-baseline justify-between gap-2 text-sm">
                      <span className={top ? "text-graphite" : "text-soft"}>{label}</span>
                      {top && <span className="font-mono text-[11px] text-blue">largest</span>}
                    </div>
                    {r && !loading && (
                      <div className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                        <Stroke value={w} chosen={top} index={i} />
                        <span className="font-mono text-sm">{fmt(w, 2)}</span>
                      </div>
                    )}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </div>
      )}

      {r && (
        <Readouts
          items={[
            ["Corruption", r.corruption.label],
            ["Largest branch", BRANCHES.find((b) => b.key === r.dominant)!.label],
            ["Inference, all four branches", ms(r.inference_ms)],
          ]}
        />
      )}
    </>
  );
}

export function SoftMoE() {
  return (
    <RestorationWorkspace<SoftResult>
      title="Soft Mixture-of-Experts Restoration"
      line="Every branch restores the image, and a gate blends them by weight."
      button="Blend and restore"
      icon={<Stack weight="light" size={20} aria-hidden />}
      call={api.soft}
      stage={(p) => <SoftStage {...p} />}
    />
  );
}
