interface Props {
  action: string
}

export function SuggestedAction({ action }: Props) {
  if (!action) return null
  return (
    <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-sm">
      <span className="text-amber-500 mt-0.5 shrink-0">→</span>
      <div className="break-words min-w-0">
        <span className="font-semibold text-amber-800">Suggested action: </span>
        <span className="text-amber-900">{action}</span>
      </div>
    </div>
  )
}
