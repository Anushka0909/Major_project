"use client"

import { create } from "zustand"
import type { AlertItem, Explainability, NewsArticle, Prediction, SimulationResult } from "@/lib/types"
import { mockAlerts, mockExplainability, mockNews, mockPredictions } from "@/lib/mock"

// Backend API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

type State = {
  sector: "pharma" | "textiles"
  month: string // YYYY-MM
  selectedPartner?: string
  predictions: Prediction[]
  alerts: AlertItem[]
  news: NewsArticle[]
  explainability?: Explainability
  loading: {
    predictions: boolean
    alerts: boolean
    news: boolean
    explainability: boolean
    simulation: boolean
  }
  error: {
    predictions?: string
    alerts?: string
    news?: string
    explainability?: string
  }
  apiConnected: boolean
  simulationResult?: SimulationResult
}

type Actions = {
  setSector: (s: "pharma" | "textiles") => void
  setMonth: (m: string) => void
  selectPartner: (countryCode: string) => void
  loadPredictions: () => Promise<void>
  loadAlerts: () => Promise<void>
  loadNews: (partner?: string) => Promise<void>
  loadExplainability: (partner?: string) => Promise<void>
  runSimulation: (target_country: string, feature: string, change_percent: number) => Promise<void>
}

export const useDashboardStore = create<State & Actions>((set, get) => ({
  sector: "pharma",
  month: "2020-01",
  predictions: [],
  alerts: [],
  news: [],
  explainability: undefined,
  loading: {
    predictions: false,
    alerts: false,
    news: false,
    explainability: false,
    simulation: false
  },
  error: {},
  apiConnected: false,
  simulationResult: undefined,

  setSector: (sector) => set({ sector }),
  setMonth: (month) => set({ month }),
  selectPartner: (selectedPartner) => set({ selectedPartner }),

  loadPredictions: async () => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, predictions: true },
      error: { ...state.error, predictions: undefined },
    }))

    try {
      // Try real API first
      const res = await fetch(
        `${API_BASE_URL}/api/predictions?sector=${sector}&month=${month}`,
        { signal: AbortSignal.timeout(8000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const predictions = await res.json()

      set((state) => ({
        predictions,
        apiConnected: true,
        loading: { ...state.loading, predictions: false },
      }))
      console.log(`✓ Loaded ${predictions.length} predictions from API`)
    } catch (error) {
      console.warn("API unavailable for predictions, using mock data:", error)
      // Fallback to mock data
      const predictions = mockPredictions({ sector, month })
      set((state) => ({
        predictions,
        apiConnected: false,
        loading: { ...state.loading, predictions: false },
      }))
    }
  },

  loadAlerts: async () => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, alerts: true },
      error: { ...state.error, alerts: undefined },
    }))

    try {
      const res = await fetch(
        `${API_BASE_URL}/api/alerts?sector=${sector}&month=${month}`,
        { signal: AbortSignal.timeout(8000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const alerts = await res.json()

      set((state) => ({
        alerts,
        loading: { ...state.loading, alerts: false },
      }))
      console.log(`✓ Loaded ${alerts.length} alerts from API`)
    } catch (error) {
      console.warn("API unavailable for alerts, using mock data:", error)
      const alerts = mockAlerts({ sector, month })
      set((state) => ({
        alerts,
        loading: { ...state.loading, alerts: false },
      }))
    }
  },

  loadNews: async (partner) => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, news: true },
      error: { ...state.error, news: undefined },
    }))

    try {
      const partnerParam = partner && partner !== "undefined" ? `&partner=${partner}` : ""
      const res = await fetch(
        `${API_BASE_URL}/api/news?sector=${sector}&month=${month}${partnerParam}`,
        { signal: AbortSignal.timeout(30000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const news = await res.json()

      set((state) => ({
        news,
        loading: { ...state.loading, news: false },
      }))
      console.log(`✓ Loaded ${news.length} articles from API`)
    } catch (error) {
      console.warn("API unavailable for news, using mock data:", error)
      const news = mockNews({ sector, month, partner })
      set((state) => ({
        news,
        loading: { ...state.loading, news: false },
      }))
    }
  },

  loadExplainability: async (partner) => {
    const { sector, month, selectedPartner } = get()
    const targetPartner = partner || selectedPartner

    if (!targetPartner || targetPartner === "undefined") {
      set({ explainability: undefined })
      return
    }

    set((state) => ({
      loading: { ...state.loading, explainability: true },
      error: { ...state.error, explainability: undefined },
    }))

    try {
      const res = await fetch(
        `${API_BASE_URL}/api/explainability?sector=${sector}&month=${month}&partner=${targetPartner}`,
        { signal: AbortSignal.timeout(8000) }
      )

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const explainability = await res.json()

      set((state) => ({
        explainability,
        loading: { ...state.loading, explainability: false },
      }))
      console.log(`✓ Loaded explainability from API for ${targetPartner}`)
    } catch (error) {
      console.warn("API unavailable for explainability, using mock data:", error)
      const explainability = mockExplainability({ sector, month, partner: targetPartner })
      set((state) => ({
        explainability,
        loading: { ...state.loading, explainability: false },
      }))
    }
  },

  runSimulation: async (targetCountry: string, feature: string, changePercent: number) => {
    const { sector, month } = get()

    set((state) => ({
      loading: { ...state.loading, simulation: true },
    }))

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_country: targetCountry,
          feature,
          change_percent: changePercent,
          sector,
          month
        })
      })

      if (!res.ok) throw new Error("Simulation failed")

      const result = await res.json()
      set({ simulationResult: result })
    } catch (error) {
      console.error("Simulation failed:", error)
      set({
        simulationResult: {
          baseline: 15000000,
          counterfactual: 12000000,
          delta: -3000000,
          pct_impact: changePercent * 0.8,
          global_impact: changePercent * 0.4,
          explanation: `Mock Simulation: A ${changePercent}% shift in ${feature} for ${targetCountry} would impact ${sector} trade patterns.`
        }
      })
    } finally {
      set((state) => ({
        loading: { ...state.loading, simulation: false },
      }))
    }
  },
}))
