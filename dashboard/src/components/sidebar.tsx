"use client"

import { useDashboardStore } from "@/components/dashboard/store"
import { MonthYearPicker } from "@/components/date/month-year-picker" // use realistic calendar

const sectors = [
  { id: "pharma", label: "Pharmaceuticals" },
  { id: "textiles", label: "Textiles & Apparel" },
] as const

export function Sidebar() {
  const { sector, setSector, month, setMonth } = useDashboardStore()

  return (
    <div className="p-4 lg:p-6 space-y-6">
      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Product sector</h2>
        <div className="grid grid-cols-1 gap-2">
          {sectors.map((s) => (
            <button
              key={s.id}
              onClick={() => setSector(s.id)}
              className={`w-full text-left px-3 py-2 rounded-md border transition ${sector === s.id ? "bg-primary text-primary-foreground" : "bg-card hover:bg-accent"
                }`}
              aria-pressed={sector === s.id}
            >
              {s.label}
            </button>
          ))}
        </div>
      </section>

      <section>
        {/* <h2 className="text-sm font-medium text-muted-foreground mb-2">Time</h2> */}
        {/* <MonthYearPicker value={month} onChange={(v) => setMonth(v)} /> */}
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Analysis</h2>
        <div className="space-y-1">
          <a href="#overview" className="block text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors">Network Overview</a>
          <a href="#predictions" className="block text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors">Trade Forecasts</a>
          <a href="#simulation" className="block text-xs px-3 py-1.5 rounded hover:bg-accent transition-colors flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-primary" />
            What-If Simulator
          </a>
        </div>
      </section>

      <section className="pt-2">
        <p className="text-xs text-muted-foreground">
          India-focused, bilateral export forecasts with transparent drivers, alerts, and news intelligence.
        </p>
      </section>
    </div>
  )
}
