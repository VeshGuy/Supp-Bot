import { useState } from 'react'
import type { Citation } from '../hooks/useChat'

interface Props {
  citation: Citation
  index: number
}

const SOURCE_STYLES: Record<string, { badge: string; label: string; linkText: string }> = {
  ticket:     { badge: 'bg-blue-100 text-blue-700',   label: '',            linkText: 'View in Zendesk →' },
  confluence: { badge: 'bg-purple-100 text-purple-700', label: 'Confluence', linkText: 'View in Confluence →' },
  jira:       { badge: 'bg-orange-100 text-orange-700', label: 'Jira',       linkText: 'View in Jira →' },
  slack:      { badge: 'bg-pink-100 text-pink-700',   label: 'Slack',       linkText: 'View in Slack →' },
  docs:       { badge: 'bg-green-100 text-green-700', label: 'Docs',        linkText: 'View in Plivo Docs →' },
}

export function CitationCard({ citation, index }: Props) {
  const [expanded, setExpanded] = useState(false)
  const isTicket = citation.source === 'ticket'
  const style = SOURCE_STYLES[citation.source] ?? SOURCE_STYLES.docs

  const title = isTicket
    ? `#${citation.ticket_id} — ${citation.subject}`
    : citation.page_title
      ? `${citation.page_title}${citation.section_title ? ` › ${citation.section_title}` : ''}`
      : citation.subject ?? 'Documentation'

  const badgeLabel = isTicket ? `[${index}]` : style.label

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden text-sm">
      <button
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 hover:bg-gray-100 text-left"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-xs font-mono px-1.5 py-0.5 rounded shrink-0 ${style.badge}`}>
            {badgeLabel}
          </span>
          <span className="font-medium text-gray-800 truncate">{title}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {isTicket && citation.csat_score && (
            <span className="text-xs text-gray-500">CSAT {citation.csat_score}</span>
          )}
          {isTicket && citation.resolution_type && (
            <span className="text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
              {citation.resolution_type.replace(/_/g, ' ')}
            </span>
          )}
          <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-3 py-2 bg-white space-y-2">
          {citation.excerpt && (
            <blockquote className="border-l-2 border-gray-300 pl-3 text-gray-600 italic">
              {citation.excerpt}
            </blockquote>
          )}
          {(citation.url || citation.zendesk_url) && (
            <a
              href={citation.url ?? citation.zendesk_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline text-xs"
            >
              {style.linkText}
            </a>
          )}
        </div>
      )}
    </div>
  )
}
