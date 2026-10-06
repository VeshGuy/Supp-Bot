import { useState } from 'react'

interface Props {
  score: number
  factors: Record<string, string | number>
}

export function ConfidenceBadge({ score, factors }: Props) {
  const [showBreakdown, setShowBreakdown] = useState(false)

  const { color, label } = score >= 0.8
    ? { color: 'bg-green-100 text-green-800 border-green-200', label: 'High' }
    : score >= 0.5
    ? { color: 'bg-yellow-100 text-yellow-800 border-yellow-200', label: 'Medium' }
    : { color: 'bg-red-100 text-red-800 border-red-200', label: 'Low' }

  const dot = score >= 0.8 ? '🟢' : score >= 0.5 ? '🟡' : '🔴'

  return (
    <div className="relative inline-block">
      <button
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color} cursor-pointer`}
        onMouseEnter={() => setShowBreakdown(true)}
        onMouseLeave={() => setShowBreakdown(false)}
      >
        <span>{dot}</span>
        <span>{label}</span>
        <span className="opacity-60">· {(score * 100).toFixed(0)}%</span>
      </button>

      {showBreakdown && Object.keys(factors).length > 0 && (
        <div className="absolute z-10 bottom-full mb-2 left-0 w-64 bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-xs">
          <div className="font-semibold text-gray-700 mb-2">Confidence breakdown</div>
          {Object.entries(factors).map(([key, val]) => (
            <div key={key} className="flex justify-between text-gray-600 py-0.5">
              <span className="capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="font-medium">{val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
