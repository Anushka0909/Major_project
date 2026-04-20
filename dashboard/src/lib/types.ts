export interface Prediction {
  partnerCode: string // ISO3 code for React key
  partner: string // Country name
  value: number
  change: number // Decimal format (0.125 = 12.5%)
  confidence: number // 0-1 scale
  risk_level: "low" | "medium" | "high"
}

export interface AlertItem {
  id: string
  type: "opportunity" | "risk"
  title: string
  summary: string
  partner: string
  partnerCode: string
  change: number
  recommendations?: Array<{
    country_code: string
    country_name: string
    predicted_value: number
    growth_rate: number
    confidence: number
    risk_level: string
    recommendation_score: number
    rationale: string
  }>
}

export interface NewsArticle {
  id: string
  title: string
  snippet: string
  source: string
  url: string
  date: string
  sentiment: number // -1.0 to 1.0
  relevance_score: number
  country_code?: string
}

export interface Explainability {
  attention: Array<{
    partner: string
    weight: number
  }>
  features: Array<{
    feature: string
    importance: number
  }>
  blurb: string
}
export interface SimulationResult {
  baseline: number
  counterfactual: number
  delta: number
  pct_impact: number
  global_impact: number
  explanation: string
}
