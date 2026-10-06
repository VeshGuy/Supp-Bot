import { useState } from 'react'
import type { Citation } from '../hooks/useChat'

interface Props {
  reasoning: string
  citations: Citation[]
}

export function ReasoningPanel({ reasoning, citations }: Props) {
  const [open, setOpen] = useState(false)

  if (!reasoning && citations.length === 0) return null

  const ticketCitations = citations.filter(c => c.source === 'ticket' && c.ticket_id)
  const docCitations = citations.filter(c => c.source !== 'ticket')

  return (
    <div className="border border-gray-100 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-500 hover:bg-gray-50 transition-colors text-left"
      >
        <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>▶</span>
        <span className="font-medium">How I answered</span>
        <span className="ml-auto text-gray-400">
          {ticketCitations.length > 0 && `${ticketCitations.length} ticket${ticketCitations.length > 1 ? 's' : ''}`}
          {docCitations.length > 0 && ` · ${docCitations.length} doc${docCitations.length > 1 ? 's' : ''}`}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 space-y-2 border-t border-gray-100 pt-2">
          {/* Reasoning */}
          {reasoning && (
            <p className="text-xs text-gray-600 leading-relaxed">{reasoning}</p>
          )}

          {/* Ticket sources */}
          {ticketCitations.length > 0 && (
            <div className="space-y-1">
              {ticketCitations.map((c, i) => (
                <div key={i} className="text-xs flex gap-2">
                  <span className="text-indigo-500 font-medium shrink-0">#{c.ticket_id}</span>
                  <span className="text-gray-500 truncate">{c.excerpt || c.subject}</span>
                </div>
              ))}
            </div>
          )}

          {/* Doc sources */}
          {docCitations.length > 0 && (
            <div className="space-y-1">
              {docCitations.map((c, i) => (
                <div key={i} className="text-xs flex gap-2">
                  <span className="text-emerald-500 font-medium shrink-0">doc</span>
                  <span className="text-gray-500 truncate">{c.page_title || c.section_title || c.url}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
