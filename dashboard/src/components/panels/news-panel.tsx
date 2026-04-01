"use client"

import type { NewsArticle } from "@/lib/types"

export function NewsPanel({ articles }: { articles: NewsArticle[] }) {
  return (
    <div className="p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">News Intelligence</h3>
        <p className="text-xs text-muted-foreground">Sentiment & sources that moved predictions</p>
      </div>

      <ul className="mt-2 divide-y border rounded-md">
        {articles.map((a) => (
          <li key={a.id} className="p-3 hover:bg-accent/60 transition">
            <a
              href={a.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block focus:outline-none focus:ring-2 focus:ring-ring rounded"
              aria-label={`Open article ${a.title} in new tab`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-medium">{a.title}</p>
                  <p className="text-xs text-muted-foreground">
                    {a.source} • {new Date(a.date).toLocaleDateString()} • Sentiment:{" "}
                    <span style={{ color: a.sentiment >= 0 ? "var(--color-chart-1)" : "var(--destructive)" }}>
                      {a.sentiment.toFixed(2)}
                    </span>
                  </p>
                  <p className="text-xs mt-1 line-clamp-2">{a.snippet}</p>
                </div>
                <span className="text-xs shrink-0">↗</span>
              </div>
            </a>
          </li>
        ))}
        {articles.length === 0 && (
          <li className="p-6 text-center text-sm text-muted-foreground">No articles for the current selection.</li>
        )}
      </ul>
    </div>
  )
}
