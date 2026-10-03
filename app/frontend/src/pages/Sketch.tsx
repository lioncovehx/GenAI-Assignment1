import { useState } from "react";
import { DownloadSimple, PencilSimpleLine } from "@phosphor-icons/react";
import { api, type SketchResult, type Source } from "../api";
import {
  btnPrimary,
  btnSecondary,
  EmptyStage,
  ErrorLine,
  Field,
  fmt,
  PageHeader,
  Plate,
  PlatePair,
  Readouts,
  SourcePicker,
  useRun,
  Workspace,
} from "../ui";

const STYLES = [1, 2, 3] as const;

export default function Sketch() {
  const [source, setSource] = useState<Source | null>(null);
  const [style, setStyle] = useState<1 | 2 | 3>(1);
  const [every, setEvery] = useState(false);
  const { data: r, error, loading, run, reset } = useRun<SketchResult>();

  return (
    <>
      <PageHeader title="Face-to-Sketch Generator" line="A conditional GAN draws a face photo as a sketch in the style you choose." />
      <Workspace
        controls={
          <>
            <SourcePicker
              kind="faces"
              webcam
              source={source}
              onChange={(s) => {
                setSource(s);
                reset();
              }}
            />
            <Field legend="Style" note="Swatches are artist sketches from the FS2K test set, one per style.">
              <div role="radiogroup" aria-label="Style" className="mt-2 grid grid-cols-3 gap-3">
                {STYLES.map((s) => (
                  <label key={s} className="group cursor-pointer has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-blue">
                    <input
                      type="radio"
                      name="style"
                      className="peer sr-only"
                      checked={style === s}
                      onChange={() => setStyle(s)}
                    />
                    <img
                      src={`/samples/style-${s}.png`}
                      alt=""
                      className="plate-img outline-offset-2 peer-checked:outline-2 peer-checked:outline-blue"
                    />
                    <span className="mt-1.5 block border-b-2 border-transparent pb-0.5 text-center text-sm text-soft peer-checked:border-blue peer-checked:text-graphite">
                      Style {s}
                    </span>
                  </label>
                ))}
              </div>
            </Field>
            <div>
              <label className="flex min-h-11 cursor-pointer items-center gap-3">
                <input type="checkbox" className="size-4" checked={every} onChange={(e) => setEvery(e.target.checked)} />
                Also show every style
              </label>
              <button
                type="button"
                className={`${btnPrimary} mt-3`}
                disabled={!source || loading}
                onClick={() => source && run(() => api.sketch(source, style, every))}
              >
                <PencilSimpleLine weight="light" size={20} aria-hidden />
                {loading ? "Working" : "Generate sketch"}
              </button>
              {r ? (
                <a href={r.sketch_png} download={`loupe-sketch-style-${r.style}.png`} className={`${btnSecondary} mt-3 w-full`}>
                  <DownloadSimple weight="light" size={20} aria-hidden /> Download PNG
                </a>
              ) : (
                <button type="button" disabled className={`${btnSecondary} mt-3 w-full`}>
                  <DownloadSimple weight="light" size={20} aria-hidden /> Download PNG
                </button>
              )}
              {!source && <p className="mt-2 text-sm text-soft">Pick a sample, upload a photo or use the webcam first.</p>}
              <ErrorLine>{error}</ErrorLine>
            </div>
          </>
        }
        stage={
          source ? (
            <section aria-label="Result" aria-busy={loading} className="sheet p-5 md:p-6">
              <PlatePair>
                <Plate src={r?.photo_png ?? source.url} caption="Photo" meta={r ? "128 x 128" : "before padding"} loading={loading} />
                <Plate
                  src={r?.sketch_png}
                  caption={r ? `Sketch, Style ${r.style}` : "Sketch"}
                  meta={r && "128 x 128"}
                  loading={loading}
                  empty="Press Generate sketch to draw it."
                />
              </PlatePair>
              {r && <Readouts items={[["Style", `Style ${r.style}`], ["Inference", `${fmt(r.inference_ms, 1)} ms`]]} />}
              {r?.all_styles && !loading && (
                <div className="mt-8">
                  <h2 className="mb-3 font-medium">Same photo, every style</h2>
                  <div className="grid grid-cols-3 gap-4">
                    {r.all_styles.map((a) => (
                      <Plate
                        key={a.style}
                        src={a.sketch_png}
                        caption={
                          <span className={`border-b-2 pb-0.5 ${a.style === r.style ? "border-blue" : "border-transparent"}`}>
                            Style {a.style}
                          </span>
                        }
                      />
                    ))}
                  </div>
                </div>
              )}
            </section>
          ) : (
            <EmptyStage />
          )
        }
      />
    </>
  );
}
