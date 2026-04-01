"use client"

import { useEffect } from "react"
import { useDashboardStore } from "./store"
import { TradeNetwork } from "../trade-network"
import { PredictionsTable } from "../predictions-table"
import { AlertsPanel } from "../panels/alerts-panel"
import { NewsPanel } from "../panels/news-panel"
import { ExplainabilityPanel } from "../panels/explainability-panel"

export default function DashboardClient() {
  const {
    sector,
    month,
    selectedPartner,
    predictions,
    alerts,
    news,
    explainability,
    apiConnected,
    selectPartner,
    loadPredictions,
    loadAlerts,
    loadNews,
    loadExplainability,
  } = useDashboardStore()

  useEffect(() => {
    loadPredictions()
    loadAlerts()
    loadNews(selectedPartner)
    loadExplainability(selectedPartner)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sector, month, selectedPartner])

  return (
    <div className="mx-auto max-w-[1400px] space-y-6">
      {/* API connection indicator */}
      <div className="flex items-center gap-2 text-xs">
        <span
          className={`size-2 rounded-full ${apiConnected ? "bg-green-500 animate-pulse" : "bg-amber-500"}`}
        />
        <span className="text-muted-foreground">
          {apiConnected ? "Live — Connected to GNN backend" : "Demo mode — Using sample data"}
        </span>
      </div>

      <section id="overview" className="grid gap-4 lg:grid-cols-5 scroll-mt-24">
        <div className="lg:col-span-3 rounded-xl border bg-card/80 shadow-sm">
          <TradeNetwork
            data={predictions}
            selectedPartner={selectedPartner}
            onSelectPartner={(cc) => selectPartner(cc)}
          />
        </div>
        <div className="lg:col-span-2 rounded-xl border bg-card/80 shadow-sm" id="alerts">
          <AlertsPanel alerts={alerts} />
        </div>
      </section>

      <section id="predictions" className="rounded-xl border bg-card/80 shadow-sm scroll-mt-24">
        <PredictionsTable
          data={predictions}
          selectedPartner={selectedPartner}
          onRowSelect={(cc) => selectPartner(cc)}
        />
      </section>

      <section id="news" className="grid gap-4 lg:grid-cols-5 scroll-mt-24">
        <div className="lg:col-span-3 rounded-xl border bg-card/80 shadow-sm">
          <NewsPanel articles={news} />
        </div>
        <div className="lg:col-span-2 rounded-xl border bg-card/80 shadow-sm" id="explainability">
          <ExplainabilityPanel explainability={explainability} />
        </div>
      </section>
    </div>
  )
}
