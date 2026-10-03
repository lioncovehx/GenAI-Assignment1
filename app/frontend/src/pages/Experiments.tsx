import { useEffect, useState } from "react";
import { ArrowSquareOut } from "@phosphor-icons/react";
import { api, type ExperimentGroup } from "../api";
import { btnSecondary, ErrorLine, PageHeader } from "../ui";

export default function Experiments() {
  const [groups, setGroups] = useState<ExperimentGroup[] | null>(null);
  const [mlflow, setMlflow] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.experiments().then(setGroups, (e: Error) => setError(e.message));
    api.experimentsMeta().then((m) => setMlflow(m.mlflow_url), () => {});
  }, []);

  return (
    <>
      <PageHeader
        title="Experiments"
        line="Logged results for the four models, read from the training records. Every run is in MLflow."
        aside={
          mlflow && (
            <a href={mlflow} target="_blank" rel="noreferrer" className={btnSecondary}>
              Open MLflow <ArrowSquareOut weight="light" size={18} aria-hidden />
            </a>
          )
        }
      />
      <ErrorLine>{error && `Could not load the results. ${error}`}</ErrorLine>
      <div className="grid gap-6 lg:grid-cols-2">
        {(groups ?? Array.from({ length: 4 }, () => null)).map((g, i) =>
          g ? (
            <section key={g.task} className="sheet p-6">
              <h2 className="mb-3 text-lg font-medium">{g.task}</h2>
              <dl className="divide-y divide-graphite/8">
                {g.facts.map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-6 py-2">
                    <dt className="text-soft">{k}</dt>
                    <dd className="text-right font-mono text-sm whitespace-nowrap">{v}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ) : (
            !error && <div key={i} className="hatch h-64" aria-label="Loading" />
          ),
        )}
      </div>
    </>
  );
}
