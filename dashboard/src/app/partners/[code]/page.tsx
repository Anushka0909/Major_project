"use client"

import Link from "next/link"
import { useEffect, useMemo } from "react"
import { useParams } from "next/navigation"
import { useDashboardStore } from "@/components/dashboard/store"
import { NewsPanel } from "@/components/panels/news-panel"
import { ExplainabilityPanel } from "@/components/panels/explainability-panel"
import { TradeNetwork } from "@/components/trade-network" // show a globe for this country

export default function PartnerDetailsPage() {
  const { code } = useParams<{ code: string }>()
  const { predictions, alerts, news, explainability, loadPredictions, loadAlerts, loadNews, loadExplainability } =
    useDashboardStore()

  useEffect(() => {
    // Load everything for this partner
    loadPredictions()
    loadAlerts()
    loadNews(code)
    loadExplainability(code)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code])

  const partnerName = useMemo(
    () => predictions.find((p) => p.partnerCode === code)?.partner || code,
    [predictions, code],
  )
  const partnerAlert = useMemo(() => alerts.find((a) => a.partnerCode === code), [alerts, code])

  return (
    <main className="min-h-dvh">
      <div className="mx-auto max-w-[1200px] px-4 lg:px-6 py-6 space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-pretty">{partnerName}</h1>
            <p className="text-sm text-muted-foreground">Detailed forecast, recommendations, news and explainability</p>
          </div>
          <Link href="/" className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground">
            Back to Dashboard
          </Link>
        </header>

        {/* Summary & Recommendations */}
        <section className="grid gap-4 md:grid-cols-5">
          <div className="md:col-span-3 rounded-lg border bg-card/80 shadow-sm p-3">
            <h2 className="text-sm font-semibold mb-2">Trade Network — Focus</h2>
            <TradeNetwork
              data={predictions.filter((p) => p.partnerCode === code)}
              selectedPartner={code}
              onSelectPartner={() => { }}
            />
          </div>
          <div className="md:col-span-2 rounded-lg border bg-card/80 shadow-sm p-4">
            <h2 className="text-sm font-semibold mb-2">Recommendations</h2>
            {partnerAlert?.recommendations?.length ? (
              <ul className="text-sm list-disc pl-5 space-y-1">
                {partnerAlert.recommendations.map((r) => (
                  <li key={r.country_code}>
                    Consider {r.country_name} — {r.rationale} (confidence {Math.round(r.confidence * 100)}%)
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">No specific recommendations.</p>
            )}
          </div>
          <div className="md:col-span-2 rounded-lg border bg-card/80 shadow-sm p-4">
            <h2 className="text-sm font-semibold mb-2">Forecast Snapshot</h2>
            <div className="text-sm space-y-1">
              {predictions
                .filter((p) => p.partnerCode === code)
                .slice(0, 1)
                .map((p) => (
                  <div key={p.partnerCode}>
                    <div>Value: {p.value.toLocaleString(undefined, { maximumFractionDigits: 0 })} USDm</div>
                    <div>
                      Change:{" "}
                      <span style={{ color: p.change >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                        {(p.change * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>Confidence: {Math.round(p.confidence * 100)}%</div>
                    <div className="text-xs text-muted-foreground mt-2">
                      Sector and time filters from dashboard apply. {/* TODO: Sync filters across routes if desired. */}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </section>

        {/* News & Explainability */}
        <section id="news" className="grid gap-4 md:grid-cols-5">
          <div className="md:col-span-3 rounded-lg border bg-card/80 shadow-sm">
            <NewsPanel articles={news} />
          </div>
          <div className="md:col-span-2 rounded-lg border bg-card/80 shadow-sm" id="explainability">
            <ExplainabilityPanel explainability={explainability} />
          </div>
        </section>
      </div>
    </main>
  )
}
