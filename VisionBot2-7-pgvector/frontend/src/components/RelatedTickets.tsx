import type { RelatedTicket } from '../hooks/useChat'

interface Props {
  tickets: RelatedTicket[]
}

export function RelatedTickets({ tickets }: Props) {
  if (!tickets?.length) return null
  return (
    <div className="mt-2">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        Related tickets
      </div>
      <div className="space-y-1">
        {tickets.map(t => (
          <div key={t.ticket_id} className="flex items-center gap-2 text-sm text-gray-600">
            <span className="text-gray-400">•</span>
            <span className="font-mono text-xs text-gray-400">#{t.ticket_id}</span>
            <span className="truncate">{t.subject}</span>
            {t.resolution_type && (
              <span className="shrink-0 text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                {t.resolution_type.replace(/_/g, ' ')}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
