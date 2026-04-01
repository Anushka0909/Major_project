"use client"

import type React from "react"

import { useMemo, useState } from "react"
import type { Prediction } from "@/lib/types"

type Props = {
  data: Prediction[]
  selectedPartner?: string
  onRowSelect: (partnerCode: string) => void
}

type SortKey = keyof Pick<Prediction, "partner" | "value" | "change" | "confidence">

export function PredictionsTable({ data, selectedPartner, onRowSelect }: Props) {
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "value", dir: "desc" })

  const sorted = useMemo(() => {
    const arr = [...data]
    arr.sort((a, b) => {
      const k = sort.key
      const va = a[k] as number | string
      const vb = b[k] as number | string
      if (typeof va === "string" && typeof vb === "string") {
        return sort.dir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va)
      }
      return sort.dir === "asc" ? (va as number) - (vb as number) : (vb as number) - (va as number)
    })
    return arr
  }, [data, sort])

  function handleSort(key: SortKey) {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }))
  }

  return (
    <div className="p-3">
      <div className="flex items-center justify-between px-2 py-2">
        <h3 className="text-sm font-semibold">Predictions</h3>
        <p className="text-xs text-muted-foreground">India → Partner; values in USD millions</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-secondary/60">
            <tr>
              <Th onClick={() => handleSort("partner")} active={sort.key === "partner"} dir={sort.dir}>
                Country
              </Th>
              <Th onClick={() => handleSort("value")} active={sort.key === "value"} dir={sort.dir}>
                Value
              </Th>
              <Th onClick={() => handleSort("change")} active={sort.key === "change"} dir={sort.dir}>
                Δ %
              </Th>
              <Th onClick={() => handleSort("confidence")} active={sort.key === "confidence"} dir={sort.dir}>
                Confidence
              </Th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-3 py-8 text-center text-muted-foreground">
                  No predictions available. Please check your data pipeline.
                </td>
              </tr>
            ) : (
              sorted.map((row) => (
                <tr
                  key={row.partnerCode}
                  className={`cursor-pointer border-b hover:bg-accent ${selectedPartner === row.partnerCode ? "bg-accent" : ""}`}
                  onClick={() => onRowSelect(row.partnerCode)}
                >
                  <td className="px-3 py-2">{row.partner}</td>
                  <td className="px-3 py-2">{row.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td className="px-3 py-2">
                    <span style={{ color: row.change >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                      {(row.change * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-3 py-2">{Math.round(row.confidence * 100)}%</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Th({
  children,
  active,
  dir,
  onClick,
}: {
  children: React.ReactNode
  active?: boolean
  dir?: "asc" | "desc"
  onClick?: () => void
}) {
  return (
    <th
      role="columnheader"
      scope="col"
      onClick={onClick}
      className="px-3 py-2 text-left font-medium select-none"
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button className="inline-flex items-center gap-1 hover:underline">
        {children}
        {active ? <span aria-hidden>{dir === "asc" ? "↑" : "↓"}</span> : null}
      </button>
    </th>
  )
}
