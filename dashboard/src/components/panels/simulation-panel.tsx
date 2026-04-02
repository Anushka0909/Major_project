"use client"

import { useState } from "react"
import { useDashboardStore } from "../dashboard/store"
import { Play, TrendingDown, TrendingUp, Info, Activity } from "lucide-react"

export function SimulationPanel() {
    const {
        selectedPartner,
        runSimulation,
        simulationResult,
        loading,
        sector
    } = useDashboardStore()

    const [change, setChange] = useState(-20)
    const [feature, setFeature] = useState("gdp")

    const handleSimulate = () => {
        if (!selectedPartner) return
        runSimulation(selectedPartner, feature, change)
    }

    if (!selectedPartner) {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-center h-full text-muted-foreground">
                <Activity className="size-12 mb-4 opacity-20" />
                <p>Select a country to run a "What-If" trade simulation</p>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full p-4 space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <Play className="size-4 text-primary" />
                    Counterfactual Simulator
                </h3>
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
                    Causal Engine
                </span>
            </div>

            <div className="space-y-4">
                <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Intervention Target
                    </label>
                    <div className="text-xl font-bold">{selectedPartner}</div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <button
                        onClick={() => setFeature("gdp")}
                        className={`px-3 py-2 rounded-lg border text-sm font-medium transition-all ${feature === "gdp"
                                ? "bg-primary text-primary-foreground border-primary"
                                : "bg-background hover:bg-muted"
                            }`}
                    >
                        GDP Shift
                    </button>
                    <button
                        onClick={() => setFeature("sentiment")}
                        className={`px-3 py-2 rounded-lg border text-sm font-medium transition-all ${feature === "sentiment"
                                ? "bg-primary text-primary-foreground border-primary"
                                : "bg-background hover:bg-muted"
                            }`}
                    >
                        Sentiment Shift
                    </button>
                </div>

                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium">Magnitude</label>
                        <span className={`text-sm font-bold ${change < 0 ? "text-red-500" : "text-green-500"}`}>
                            {change > 0 ? "+" : ""}{change}%
                        </span>
                    </div>
                    <input
                        type="range"
                        min="-50"
                        max="50"
                        step="5"
                        value={change}
                        onChange={(e) => setChange(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                    <div className="flex justify-between text-[10px] text-muted-foreground font-medium">
                        <span>CRASH (-50%)</span>
                        <span>NEUTRAL</span>
                        <span>BOOM (+50%)</span>
                    </div>
                </div>

                <button
                    onClick={handleSimulate}
                    disabled={loading.simulation}
                    className="w-full py-3 rounded-xl bg-primary text-primary-foreground font-bold shadow-lg shadow-primary/20 hover:scale-[1.02] transition-transform flex items-center justify-center gap-2 disabled:opacity-50 disabled:scale-100"
                >
                    {loading.simulation ? (
                        <div className="size-4 border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin rounded-full" />
                    ) : (
                        <Activity className="size-4" />
                    )}
                    Run Simulation
                </button>
            </div>

            {simulationResult && (
                <div className="mt-4 pt-4 border-t space-y-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 rounded-lg bg-muted/30 border">
                            <div className="text-[10px] text-muted-foreground uppercase font-bold mb-1">Baseline</div>
                            <div className="text-lg font-bold">
                                ${(simulationResult.baseline / 1000000).toFixed(1)}M
                            </div>
                        </div>
                        <div className="p-3 rounded-lg bg-primary/5 border border-primary/20">
                            <div className="text-[10px] text-primary uppercase font-bold mb-1">Simulated</div>
                            <div className="text-lg font-bold text-primary">
                                ${(simulationResult.counterfactual / 1000000).toFixed(1)}M
                            </div>
                        </div>
                    </div>

                    <div className="p-4 rounded-xl bg-muted/20 border-l-4 border-primary">
                        <div className="flex items-start gap-3">
                            <Info className="size-4 text-primary mt-1 shrink-0" />
                            <div className="space-y-1">
                                <div className="text-xs font-bold text-foreground">Economic Impact</div>
                                <p className="text-sm text-muted-foreground leading-relaxed">
                                    {simulationResult.explanation}
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                        {simulationResult.pct_impact < 0 ? (
                            <TrendingDown className="size-3 text-red-500" />
                        ) : (
                            <TrendingUp className="size-3 text-green-500" />
                        )}
                        Global ripple effect: {simulationResult.global_impact.toFixed(2)}%
                    </div>
                </div>
            )}
        </div>
    )
}
