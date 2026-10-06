import ReactMarkdown from 'react-markdown'
import { useState, memo } from 'react'
import type { Citation, Message as MessageType } from '../hooks/useChat'
import { CitationCard } from './CitationCard'
import { ConfidenceBadge } from './ConfidenceBadge'
import { SuggestedAction } from './SuggestedAction'
import { RelatedTickets } from './RelatedTickets'
import { ReasoningPanel } from './ReasoningPanel'

const INITIAL_SHOW = 3

function CitationList({ citations }: { citations: Citation[] }) {
  const [showAll, setShowAll] = useState(false)
  const visible = showAll ? citations : citations.slice(0, INITIAL_SHOW)
  const hidden = citations.length - INITIAL_SHOW

  return (
    <div className="space-y-1.5">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Sources</div>
      {visible.map((c, i) => (
        <CitationCard key={c.ticket_id ?? `doc-${i}`} citation={c} index={i + 1} />
      ))}
      {!showAll && hidden > 0 && (
        <button
          onClick={() => setShowAll(true)}
          className="text-xs text-indigo-600 hover:text-indigo-800 font-medium pt-0.5"
        >
          + {hidden} more source{hidden > 1 ? 's' : ''}
        </button>
      )}
      {showAll && citations.length > INITIAL_SHOW && (
        <button
          onClick={() => setShowAll(false)}
          className="text-xs text-gray-400 hover:text-gray-600 pt-0.5"
        >
          Show less
        </button>
      )}
    </div>
  )
}

interface Props {
  message: MessageType
  onFeedback?: (msgId: string, rating: number, comment?: string, issueType?: string) => void
}

const FEEDBACK_KEY = (id: string) => `feedback_submitted_${id}`

const ISSUE_TYPES = [
  { value: 'wrong_topic',      label: 'Wrong topic — missed what I was asking' },
  { value: 'wrong_resolution', label: 'Right topic but wrong or outdated resolution' },
  { value: 'too_vague',        label: 'Correct diagnosis but steps were too vague' },
  { value: 'wrong_facts',      label: 'Ryuk paths or facts were incorrect' },
  { value: 'missing_info',     label: 'Missing info I expected it to know' },
]

function FeedbackWidget({ messageId, onFeedback }: {
  messageId: string
  onFeedback: (msgId: string, rating: number, comment?: string, issueType?: string) => void
}) {
  const [selected, setSelected] = useState<number | null>(null)
  const [comment, setComment] = useState('')
  const [issueType, setIssueType] = useState('')
  const [submitted, setSubmitted] = useState(() => !!localStorage.getItem(FEEDBACK_KEY(messageId)))

  if (submitted) {
    return <p className="text-xs text-gray-400 pt-1">Thanks for your feedback!</p>
  }

  const handleSubmit = () => {
    if (selected === null) return
    onFeedback(messageId, selected, comment.trim() || undefined, issueType || undefined)
    localStorage.setItem(FEEDBACK_KEY(messageId), '1')
    setSubmitted(true)
  }

  const isHigh   = selected !== null && selected >= 7
  const isMedium = selected !== null && selected >= 5 && selected <= 6
  const isLow    = selected !== null && selected <= 4

  return (
    <div className="pt-1 space-y-2 min-h-[2rem]">
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-xs text-gray-400 mr-1">Rate this answer:</span>
        {Array.from({ length: 10 }, (_, i) => i + 1).map(n => (
          <button
            key={n}
            onClick={() => setSelected(n)}
            className={`w-6 h-6 text-xs rounded-md font-medium transition-colors ${
              selected === n
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-indigo-100 hover:text-indigo-700'
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {isHigh && (
        <div className="flex gap-2 items-start">
          <input
            type="text"
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="What was most useful? (optional)"
            className="flex-1 text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-400"
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          />
          <button onClick={handleSubmit} className="text-xs bg-indigo-600 text-white rounded-lg px-3 py-1.5 hover:bg-indigo-700 transition-colors shrink-0">
            Submit
          </button>
        </div>
      )}

      {isMedium && (
        <div className="flex gap-2 items-start">
          <input
            type="text"
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="What could be improved? (optional)"
            className="flex-1 text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-400"
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          />
          <button onClick={handleSubmit} className="text-xs bg-indigo-600 text-white rounded-lg px-3 py-1.5 hover:bg-indigo-700 transition-colors shrink-0">
            Submit
          </button>
        </div>
      )}

      {isLow && (
        <div className="space-y-2">
          <select
            value={issueType}
            onChange={e => setIssueType(e.target.value)}
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-400 bg-white text-gray-700"
          >
            <option value="">What went wrong? (optional)</option>
            {ISSUE_TYPES.map(t => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
          <div className="flex gap-2 items-start">
            <input
              type="text"
              value={comment}
              onChange={e => setComment(e.target.value)}
              placeholder="Any additional details? (optional)"
              className="flex-1 text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-400"
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            />
            <button onClick={handleSubmit} className="text-xs bg-indigo-600 text-white rounded-lg px-3 py-1.5 hover:bg-indigo-700 transition-colors shrink-0">
              Submit
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export const Message = memo(function Message({ message, onFeedback }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-xl bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
          {message.content}
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex gap-3 max-w-3xl">
      <div className="shrink-0 w-7 h-7 bg-indigo-100 rounded-full flex items-center justify-center text-sm mt-0.5">
        🤖
      </div>
      <div className="flex-1 min-w-0 space-y-3">
        {/* Loading indicator */}
        {message.loading && !message.content && (
          <div className="flex gap-1 items-center text-gray-400 text-sm py-1">
            <span className="animate-bounce">●</span>
            <span className="animate-bounce [animation-delay:0.1s]">●</span>
            <span className="animate-bounce [animation-delay:0.2s]">●</span>
          </div>
        )}

        {/* Answer */}
        {message.content && (
          <div className="prose prose-sm max-w-none text-gray-800">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Confidence + Suggested Action */}
        {!message.loading && message.confidence_score !== undefined && (
          <div className="flex flex-wrap items-start gap-2">
            <ConfidenceBadge
              score={message.confidence_score}
              factors={message.confidence_factors ?? {}}
            />
          </div>
        )}

        {message.suggested_action && (
          <SuggestedAction action={message.suggested_action} />
        )}

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <CitationList citations={message.citations} />
        )}

        {/* Related Tickets */}
        {message.related_tickets && message.related_tickets.length > 0 && (
          <RelatedTickets tickets={message.related_tickets} />
        )}

        {/* Reasoning panel */}
        {!message.loading && (message.reasoning || (message.citations && message.citations.length > 0)) && (
          <ReasoningPanel
            reasoning={message.reasoning ?? ''}
            citations={message.citations ?? []}
          />
        )}

        {/* Feedback */}
        {!message.loading && onFeedback && (
          <FeedbackWidget messageId={message.id} onFeedback={onFeedback} />
        )}
      </div>
    </div>
  )
})
