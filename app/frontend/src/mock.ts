// Mock backend for UI work (VITE_MOCK=1). No models run here: images are canvas transforms and all numbers are fake.

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const LEVEL = { low: 0, medium: 1, high: 2 } as const;
type Sev = keyof typeof LEVEL;

// Seeded PRNG so mock corruptions look the same on every run (stable screenshots).
function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

async function loadSource(f: FormData): Promise<ImageBitmap> {
  const id = f.get("sample_id");
  const file = f.get("image");
  if (!id && !(file instanceof Blob)) throw new Error("Send an image or a sample_id.");
  const blob = id ? await (await fetch(`/samples/${id}.png`)).blob() : (file as Blob);
  try {
    return await createImageBitmap(blob);
  } catch {
    throw new Error("This file is not an image. Choose a PNG or JPG.");
  }
}

function canvas(src: CanvasImageSource & { width: number; height: number }, filter = "none") {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d")!;
  ctx.filter = filter;
  const s = Math.min(src.width, src.height); // centre crop to square, then 128 x 128
  ctx.drawImage(src, (src.width - s) / 2, (src.height - s) / 2, s, s, 0, 0, 128, 128);
  return c;
}

function corrupt(clean: HTMLCanvasElement, type: string, sev: Sev) {
  const i = LEVEL[sev];
  if (type === "blur") return canvas(clean, `blur(${[0.7, 1.5, 2.5][i]}px)`);
  const c = canvas(clean);
  const ctx = c.getContext("2d")!;
  const rand = rng(7 + i);
  if (type === "salt") {
    const img = ctx.getImageData(0, 0, 128, 128);
    const p = [0.03, 0.08, 0.15][i];
    for (let px = 0; px < 128 * 128; px++) {
      if (rand() < p) {
        const v = rand() < 0.5 ? 0 : 255;
        img.data.set([v, v, v], px * 4);
      }
    }
    ctx.putImageData(img, 0, 0);
  } else if (type === "occlusion") {
    const boxes = [1, 2, 3][i];
    const side = Math.sqrt(([0.1, 0.2, 0.35][i] * 128 * 128) / boxes);
    ctx.fillStyle = "#000";
    for (let b = 0; b < boxes; b++) ctx.fillRect(rand() * (128 - side), rand() * (128 - side), side, side);
  }
  return c;
}

const LABELS: Record<string, (s: Sev) => string> = {
  none: () => "None, image used as uploaded",
  salt: (s) => `Salt-and-pepper, ${s}, p = ${[0.03, 0.08, 0.15][LEVEL[s]]}`,
  blur: (s) => `Gaussian blur, ${s}, kernel ${[3, 5, 7][LEVEL[s]]}, sigma ${[0.7, 1.5, 2.5][LEVEL[s]]}`,
  occlusion: (s) =>
    `Occlusion, ${s}, ${[1, 2, 3][LEVEL[s]]} box${s === "low" ? "" : "es"}, ${[10, 20, 35][LEVEL[s]]}% of the image`,
};

// Example routing numbers per corruption (not model output).
const EXAMPLE_PROBS: Record<string, [number, number, number, number]> = {
  none: [0.981, 0.002, 0.015, 0.002],
  salt: [0.001, 0.996, 0.002, 0.001],
  blur: [0.004, 0.001, 0.993, 0.002],
  occlusion: [0.006, 0.002, 0.004, 0.988],
};
const EXAMPLE_WEIGHTS: Record<string, [number, number, number, number]> = {
  none: [0.98, 0.0, 0.01, 0.01],
  salt: [0.01, 0.97, 0.01, 0.01],
  blur: [0.58, 0.0, 0.41, 0.01],
  occlusion: [0.2, 0.01, 0.02, 0.77],
};
const KEYS = ["identity", "salt", "blur", "occlusion"] as const;
const CLASSES = ["clean", "salt", "blur", "occlusion"] as const;

async function restore(kind: string, f: FormData) {
  const type = (f.get("corruption") as string) || "none";
  const sev = ((f.get("severity") as string) || "medium") as Sev;
  const src = canvas(await loadSource(f));
  const input = type === "none" ? src : corrupt(src, type, sev);
  const restored = canvas(src, "blur(0.5px)"); // mock "restoration": the clean source, slightly softened
  await sleep(600);
  const shared = {
    input_png: input.toDataURL(),
    output_png: restored.toDataURL(),
    corruption: { type, severity: sev, label: LABELS[type](sev) },
  };
  if (kind === "universal") return { ...shared, inference_ms: 3.4 };
  if (kind === "hard") {
    const p = EXAMPLE_PROBS[type];
    const predicted = CLASSES[p.indexOf(Math.max(...p))];
    return {
      ...shared,
      output_png: predicted === "clean" ? input.toDataURL() : shared.output_png,
      probs: Object.fromEntries(CLASSES.map((k, i) => [k, p[i]])),
      predicted,
      expert: predicted === "clean" ? "identity" : predicted,
      inference_ms: { classifier: 0.4, expert: 3.1, total: 3.5 },
    };
  }
  const w = EXAMPLE_WEIGHTS[type];
  const branchFilters = ["none", "blur(0.8px)", "contrast(1.15) saturate(1.1)", "blur(1.2px)"];
  return {
    ...shared,
    weights: Object.fromEntries(KEYS.map((k, i) => [k, w[i]])),
    dominant: KEYS[w.indexOf(Math.max(...w))],
    branch_pngs: Object.fromEntries(KEYS.map((k, i) => [k, canvas(input, branchFilters[i]).toDataURL()])),
    inference_ms: 14.2,
  };
}

async function sketch(f: FormData) {
  const style = Number(f.get("style") || 1) as 1 | 2 | 3;
  const photo = canvas(await loadSource(f));
  // mock "sketch": grey, high contrast, a different tone per style
  const draw = (s: number) =>
    canvas(photo, ["grayscale(1) contrast(1.9) brightness(1.45)", "grayscale(1) contrast(2.6) brightness(1.05)", "grayscale(1) contrast(1.4) brightness(1.2)"][s - 1]).toDataURL();
  await sleep(600);
  return {
    photo_png: photo.toDataURL(),
    sketch_png: draw(style),
    style,
    inference_ms: 3.3,
    ...(f.get("all_styles") === "true" && {
      all_styles: ([1, 2, 3] as const).map((s) => ({ style: s, sketch_png: draw(s) })),
    }),
  };
}

const range = (n: number) => Array.from({ length: n }, (_, i) => String(i + 1).padStart(2, "0"));

export async function mockFetch(path: string, init?: RequestInit): Promise<unknown> {
  const body = init?.body as FormData | undefined;
  if (path === "/health") return { status: "ok", models: { universal: true, hard_routing: true, soft_moe: true, sketch: true } };
  if (path.startsWith("/samples")) {
    const kind = path.endsWith("faces") ? "faces" : "pets";
    return range(6).map((n) => ({
      id: `${kind}-${n}`,
      url: `/samples/${kind}-${n}.png`,
      label: kind === "pets" ? `Oxford pet test image ${Number(n)}` : `FS2K test photo ${Number(n)}`,
    }));
  }
  if (path.startsWith("/restore/")) return restore(path.split("/")[2], body!);
  if (path === "/sketch") return sketch(body!);
  if (path === "/experiments/meta") return { mlflow_url: "http://localhost:5001" };
  if (path === "/experiments")
    // Real logged numbers (test set unless marked val). The backend serves its own copy.
    return [
      { task: "Universal autoencoder", facts: [
        ["Test PSNR, salt-and-pepper", "27.62 dB"], ["Test PSNR, Gaussian blur", "26.94 dB"],
        ["Test PSNR, occlusion", "22.15 dB"], ["Test SSIM, clean photos", "0.854"],
        ["ONNX size", "17.9 MB"], ["CPU time per image", "3.4 ms"]] },
      { task: "Hard routing", facts: [
        ["Classifier test accuracy", "0.9963"], ["Classifier macro-F1", "0.9940"],
        ["Clean photos returned untouched", "99.16%"], ["Test PSNR, salt-and-pepper", "28.45 dB"],
        ["Test PSNR, Gaussian blur", "27.47 dB"], ["Test PSNR, occlusion", "22.66 dB"],
        ["Classifier ONNX size", "2.2 MB"]] },
      { task: "Soft mixture of experts", facts: [
        ["Val PSNR, salt-and-pepper", "29.00 dB"], ["Val PSNR, Gaussian blur", "30.50 dB"],
        ["Val PSNR, occlusion", "21.55 dB"], ["Gate accuracy (val)", "0.806"],
        ["Identity weight on blurred images (val)", "0.59"]] },
      { task: "Face-to-sketch generator", facts: [
        ["Test LPIPS", "0.207"], ["Test LPIPS, pix2pix defaults", "0.222"], ["Test SSIM", "0.525"],
        ["ONNX size", "94.3 MB"], ["CPU time per image", "3.3 ms"]] },
    ];
  throw new Error(`Mock has no answer for ${path}`);
}
