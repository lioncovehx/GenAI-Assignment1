import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "@phosphor-icons/react";
import { api } from "../api";
import { Wipe, btnPrimary } from "../ui";

type Pair = [string, string];
const sample = (id: string) => ({ kind: "sample" as const, id, url: `/samples/${id}.png` });

// Static images stay up while the live results load, and stay for good if the backend is offline.
const TILES = [
  {
    to: "/universal",
    title: "Universal Restoration",
    line: "One autoencoder repairs noise, blur and occlusion.",
    images: ["/samples/pets-03.png"],
    labels: ["Salt-and-pepper, medium", "Restored"],
    span: "md:col-span-7",
    live: () => api.universal(sample("pets-03"), "salt", "medium").then((r): Pair => [r.input_png, r.output_png]),
  },
  {
    to: "/hard",
    title: "Hard-Routed Restoration",
    line: "A classifier picks one specialist.",
    images: ["/samples/pets-05.png"],
    labels: ["Blur, medium", "Restored"],
    span: "md:col-span-5",
    live: () => api.hard(sample("pets-05"), "blur", "medium").then((r): Pair => [r.input_png, r.output_png]),
  },
  {
    to: "/soft",
    title: "Soft Mixture-of-Experts Restoration",
    line: "Four branches, blended by learned weights.",
    images: ["/samples/pets-02.png"],
    labels: ["Blur, low", "Restored"],
    span: "md:col-span-5",
    live: () => api.soft(sample("pets-02"), "blur", "low").then((r): Pair => [r.input_png, r.output_png]),
  },
  {
    to: "/sketch",
    title: "Face-to-Sketch Generator",
    line: "Turn a face photo into a sketch in one of three styles.",
    images: ["/samples/home-photo.png", "/samples/home-sketch.png"],
    labels: ["Photo", "Sketch, style 2"],
    span: "md:col-span-7",
    live: () => api.sketch(sample("faces-01"), 2, false).then((r): Pair => [r.photo_png, r.sketch_png]),
  },
];

const IMG = "plate-img w-[clamp(8rem,44%,14rem)]";

export default function Home() {
  // undefined = loading, null = backend unreachable
  const [live, setLive] = useState<(Pair | null | undefined)[]>(TILES.map(() => undefined));
  useEffect(() => {
    let alive = true;
    TILES.forEach((t, i) =>
      t.live().then(
        (pair) => alive && setLive((l) => l.map((v, j) => (j === i ? pair : v))),
        () => alive && setLive((l) => l.map((v, j) => (j === i ? null : v))),
      ),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <>
      {/* Hero: live before/after wipe */}
      <header className="mb-14 grid items-center gap-10 lg:mt-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,30rem)]">
        <div>
          <h1 className="font-display text-[clamp(3.25rem,8vw,6rem)] leading-[0.95] font-bold tracking-[-0.04em]">Loupe</h1>
          <p className="mt-5 max-w-[34ch] text-xl leading-snug text-soft">
            Four image models. Upload a picture, see what each one does and why.
          </p>
          <Link to="/universal" className={`${btnPrimary.replace("w-full ", "")} mt-8 px-6`}>
            Try a restoration
            <ArrowRight weight="light" size={18} aria-hidden />
          </Link>
        </div>
        <div className="sheet p-5">
          {live[0] ? (
            <Wipe before={live[0][0]} after={live[0][1]} />
          ) : (
            <div className="registered">
              <div className="hatch aspect-square" aria-label={live[0] === null ? "Backend offline" : "Loading example"} />
            </div>
          )}
          <p className="mt-3 text-sm text-soft">
            {live[0] === null
              ? "Start the backend to see a live example."
              : "A test cat with salt-and-pepper noise, restored by the universal autoencoder. Drag the blue line."}
          </p>
        </div>
      </header>
      <ul className="grid gap-6 md:grid-cols-12">
        {TILES.map((t, i) => {
          const pair = live[i];
          return (
            <li key={t.to} className={t.span}>
              <Link
                to={t.to}
                className="group sheet flex h-full flex-col gap-6 p-6 transition-[translate] duration-200 ease-out hover:-translate-y-0.5"
              >
                <div>
                  <h2 className="text-lg font-medium">{t.title}</h2>
                  <p className="mt-1 text-soft">{t.line}</p>
                </div>
                <div className="flex gap-4">
                  {pair ? (
                    pair.map((src, k) => (
                      <figure key={k} className="w-[clamp(8rem,44%,14rem)]">
                        <img src={src} alt={t.labels[k]} className="plate-img" />
                        <figcaption className="mt-2 text-sm text-soft">{t.labels[k]}</figcaption>
                      </figure>
                    ))
                  ) : (
                    <>
                      {t.images.map((src) => (
                        <img key={src} src={src} alt="" className={IMG} />
                      ))}
                      {pair === undefined && t.images.length === 1 && (
                        <div className={`hatch aspect-square ${IMG}`} aria-label="Loading result" />
                      )}
                    </>
                  )}
                </div>
                <span className="mt-auto inline-flex items-center gap-2 self-end font-medium">
                  Open
                  <ArrowRight weight="light" size={18} className="transition-transform duration-200 group-hover:translate-x-0.5" aria-hidden />
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </>
  );
}
