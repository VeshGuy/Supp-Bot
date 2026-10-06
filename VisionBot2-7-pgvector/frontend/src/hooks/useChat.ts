import { useState, useCallback, useEffect } from 'react'
import { MESSAGES_KEY } from './useSession'

const API_URL = '/api'
const TOKEN = import.meta.env.VITE_API_TOKEN ?? ''

export interface Citation {
  source: 'ticket' | 'docs'
  ticket_id?: string
  subject?: string
  resolution_type?: string
  csat_score?: number
  zendesk_url?: string
  page_title?: string
  section_title?: string
  url?: string
  excerpt?: string
}

export interface RelatedTicket {
  ticket_id: string
  subject: string
  resolution_type?: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  citations?: Citation[]
  confidence_score?: number
  confidence_factors?: Record<string, string | number>
  suggested_action?: string
  related_tickets?: RelatedTicket[]
  awaiting_clarification?: boolean
  loading?: boolean
}

export function useChat(sessionId: string, onFirstMessage?: (msg: string) => void) {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const stored = localStorage.getItem(MESSAGES_KEY(sessionId))
      if (stored) return (JSON.parse(stored) as Message[]).filter(m => !m.loading)
    } catch {}
    return []
  })
  const [isStreaming, setIsStreaming] = useState(false)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(MESSAGES_KEY(sessionId))
      if (stored) {
        setMessages((JSON.parse(stored) as Message[]).filter(m => !m.loading))
      } else {
        setMessages([])
      }
    } catch {
      setMessages([])
    }
  }, [sessionId])

  // Intentionally omit sessionId from deps — save ONLY when messages change.
  // Including sessionId causes this to fire with stale (previous session's) messages
  // the moment the session switches, corrupting the new session's localStorage entry.
  // By the time messages change (load effect has run), sessionId is already correct.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const toSave = messages.filter(m => !m.loading)
    if (toSave.length > 0) {
      localStorage.setItem(MESSAGES_KEY(sessionId), JSON.stringify(toSave))
    }
  }, [messages])

  const sendMessage = useCallback(async (question: string) => {
    if (isStreaming) return

    // Register session title on first message
    if (messages.length === 0 && onFirstMessage) {
      onFirstMessage(question)
    }

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: question }
    const botId = crypto.randomUUID()
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }

    setMessages(prev => [...prev, userMsg, botMsg])
    setIsStreaming(true)

    try {
      // Send the full conversation history so the backend always has complete context.
      // Filter out loading placeholders; only include role + content for the backend.
      const priorContext = messages
        .filter(m => !m.loading)
        .map(m => ({ role: m.role, content: m.content }))

      const resp = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${TOKEN}`,
        },
        body: JSON.stringify({ question, session_id: sessionId, metadata_filters: {}, prior_context: priorContext }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let content = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = JSON.parse(line.slice(6))

          if (payload.type === 'token') {
            content += payload.content
            setMessages(prev =>
              prev.map(m => m.id === botId ? { ...m, content, loading: false } : m)
            )
          } else if (payload.type === 'done') {
            setMessages(prev =>
              prev.map(m =>
                m.id === botId
                  ? {
                      ...m,
                      content,
                      loading: false,
                      reasoning: payload.reasoning ?? '',
                      citations: payload.citations ?? [],
                      confidence_score: payload.confidence_score,
                      confidence_factors: payload.confidence_factors,
                      suggested_action: payload.suggested_action,
                      related_tickets: payload.related_tickets ?? [],
                      awaiting_clarification: payload.awaiting_clarification,
                    }
                  : m
              )
            )
          } else if (payload.type === 'error') {
            setMessages(prev =>
              prev.map(m =>
                m.id === botId
                  ? { ...m, content: 'Something went wrong. Please try again.', loading: false }
                  : m
              )
            )
          }
        }
      }
    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === botId
            ? { ...m, content: 'Connection error. Please try again.', loading: false }
            : m
        )
      )
    } finally {
      setIsStreaming(false)
    }
  }, [isStreaming, sessionId, messages.length, onFirstMessage])

  const sendFeedback = useCallback(async (question: string, rating: number, ticketIds: string[], comment?: string, issueType?: string) => {
    await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${TOKEN}` },
      body: JSON.stringify({ session_id: sessionId, question, rating, comment, issue_type: issueType, ticket_ids: ticketIds }),
    })
  }, [sessionId])

  return { messages, isStreaming, sendMessage, sendFeedback }
}
