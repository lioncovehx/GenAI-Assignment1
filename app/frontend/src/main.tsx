import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { createHashRouter, Link, NavLink, Outlet, RouterProvider, useLocation } from "react-router-dom";
import { CheckCircle, Flask, GitFork, List, MagicWand, MagnifyingGlass, PencilSimpleLine, Stack, WarningCircle, X } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./index.css";
import { api, type Health } from "./api";
import Home from "./pages/Home";
import { HardRouted, SoftMoE, Universal } from "./pages/Restoration";
import Sketch from "./pages/Sketch";
import Experiments from "./pages/Experiments";

const NAV = [
  { to: "/universal", label: "Universal Restoration", icon: MagicWand },
  { to: "/hard", label: "Hard-Routed Restoration", icon: GitFork },
  { to: "/soft", label: "Soft Mixture-of-Experts Restoration", icon: Stack },
  { to: "/sketch", label: "Face-to-Sketch Generator", icon: PencilSimpleLine },
  { to: "/experiments", label: "Experiments", icon: Flask },
] as const;

function Status() {
  const [health, setHealth] = useState<Health | null | "offline">(null);
  useEffect(() => {
    const check = () => api.health().then(setHealth, () => setHealth("offline"));
    check();
    const t = setInterval(check, 15000);
    return () => clearInterval(t);
  }, []);
  if (health === null) return <p className="text-sm text-soft">Checking models</p>;
  if (health === "offline" || health.status !== "ok") {
    const down = health === "offline" ? [] : Object.keys(health.models).filter((k) => !health.models[k]);
    return (
      <p role="status" className="flex items-center gap-2 text-sm text-brick">
        <WarningCircle weight="light" size={18} aria-hidden />
        {down.length ? `Not loaded: ${down.join(", ").replaceAll("_", " ")}` : "Backend offline"}
      </p>
    );
  }
  return (
    <p role="status" className="flex items-center gap-2 text-sm">
      <CheckCircle weight="light" size={18} aria-hidden /> Models ready
    </p>
  );
}

function NavList({ onPick, id }: { onPick?: () => void; id: string }) {
  const reduce = useReducedMotion();
  return (
    <ul className="space-y-0.5">
      {NAV.map(({ to, label, icon: Icon }) => (
        <li key={to}>
          <NavLink
            to={to}
            onClick={onPick}
            className={({ isActive }) =>
              `relative flex min-h-11 items-center gap-3 rounded-[6px] py-2 pr-1 pl-3.5 text-[15px] leading-snug transition-colors duration-150 ${
                isActive ? "text-graphite" : "text-soft hover:bg-graphite/[0.04] hover:text-graphite"
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  // the active marker and its soft sheet glide between items instead of jumping
                  <motion.span
                    layoutId={`rail-${id}`}
                    className="absolute inset-0 rounded-[6px] bg-sheet/60 shadow-[inset_0_0_0_1px_rgb(42_44_47/0.08)]"
                    transition={reduce ? { duration: 0 } : { type: "spring", duration: 0.35, bounce: 0.12 }}
                    aria-hidden
                  />
                )}
                <Icon weight="light" size={20} className="relative shrink-0" aria-hidden />
                <span className="relative">{label}</span>
              </>
            )}
          </NavLink>
        </li>
      ))}
    </ul>
  );
}

function Shell() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);
  const wordmark = (
    <Link to="/" className="inline-flex items-center gap-2 font-display text-[1.75rem] leading-none font-bold tracking-[-0.03em]">
      <MagnifyingGlass weight="light" size={26} className="-scale-x-100" aria-hidden />
      Loupe
    </Link>
  );
  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
      {/* Rail: fixed on wide screens, a top bar with a menu under 1024px */}
      <nav aria-label="Workspaces" className="hidden lg:block">
        <div className="sticky top-0 flex h-dvh flex-col border-r border-graphite/12 px-3 py-8">
          <div className="px-4">{wordmark}</div>
          <div className="mt-10 flex-1">
            <NavList id="desktop" />
          </div>
          <div className="border-t border-graphite/12 px-4 pt-4">
            <Status />
          </div>
        </div>
      </nav>
      <div className="sticky top-0 z-10 border-b border-graphite/12 bg-paper lg:hidden">
        <div className="flex items-center justify-between gap-4 px-4 py-2 md:px-8">
          {wordmark}
          <div className="flex items-center gap-4">
            <Status />
            <button
              type="button"
              aria-expanded={open}
              aria-controls="mobile-nav"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen(!open)}
              className="grid size-11 place-items-center rounded-[6px]"
            >
              {open ? <X weight="light" size={24} /> : <List weight="light" size={24} />}
            </button>
          </div>
        </div>
        {open && (
          <nav id="mobile-nav" aria-label="Workspaces" className="border-t border-graphite/12 px-2 pb-3 md:px-6">
            <NavList id="mobile" onPick={() => setOpen(false)} />
          </nav>
        )}
      </div>
      <main className="mx-auto w-full max-w-[1400px] px-4 py-8 md:px-8 lg:px-10 lg:py-10">
        <Outlet />
      </main>
    </div>
  );
}

const router = createHashRouter([
  {
    element: <Shell />,
    children: [
      { path: "/", element: <Home /> },
      { path: "/universal", element: <Universal /> },
      { path: "/hard", element: <HardRouted /> },
      { path: "/soft", element: <SoftMoE /> },
      { path: "/sketch", element: <Sketch /> },
      { path: "/experiments", element: <Experiments /> },
      { path: "*", element: <Home /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
