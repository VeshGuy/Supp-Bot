import { useRef, useEffect, useState, useCallback } from 'react'
import { Message } from './components/Message'
import { Sidebar } from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useSession } from './hooks/useSession'

const EXAMPLE_QUERIES = [
  'Why are DTMF tones not detected during IVR navigation?',
  'SIP registration failing with 403 Forbidden',
  'WebRTC call drops after 30 seconds',
  'Calls to China failing with Interconnect Error',
]

export default function App() {
  const { sessionId, sessions, updateSessionTitle, resetSession, switchSession, deleteSession } = useSession()
  const onFirstMessage = useCallback((msg: string) => updateSessionTitle(sessionId, msg), [sessionId, updateSessionTitle])
  const { messages, isStreaming, sendMessage, sendFeedback } = useChat(sessionId, onFirstMessage)
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Keep a ref to messages so handleFeedback can access the latest without being recreated
  const messagesRef = useRef(messages)
  messagesRef.current = messages

  // Stable feedback handler — same reference across renders, Message memo works correctly
  const handleFeedback = useCallback((msgId: string, rating: number, comment?: string) => {
    const msgs = messagesRef.current
    const idx = msgs.findIndex(m => m.id === msgId)
    const userMsg = msgs[idx - 1]
    const msg = msgs[idx]
    const ticketIds = msg?.citations?.flatMap(c => c.ticket_id ? [c.ticket_id] : []) ?? []
    sendFeedback(userMsg?.content ?? '', rating, ticketIds, comment)
  }, [sendFeedback])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Clear input when switching sessions
  useEffect(() => { setInput('') }, [sessionId])

  const handleSend = () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')
    sendMessage(q)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        currentSessionId={sessionId}
        onSelect={switchSession}
        onNewChat={resetSession}
        onDelete={deleteSession}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.length === 0 && (
              <div className="text-center py-16">
                <div className="text-4xl mb-4">🔍</div>
                <h1 className="text-2xl font-bold text-gray-800 mb-2">SupportIQ</h1>
                <p className="text-gray-500 mb-8 max-w-md mx-auto">
                  Ask anything about past resolved support tickets. Every answer is cited.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto">
                  {EXAMPLE_QUERIES.map(q => (
                    <button
                      key={q}
                      onClick={() => { setInput(q); sendMessage(q) }}
                      className="text-left text-sm bg-white border border-gray-200 rounded-xl px-4 py-3 hover:border-indigo-300 hover:bg-indigo-50 transition-colors text-gray-700"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <Message
                key={msg.id}
                message={msg}
                onFeedback={msg.role === 'assistant' && !msg.loading ? handleFeedback : undefined}
              />
            ))}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-4 py-4">
          <div className="max-w-3xl mx-auto flex gap-3">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about a voice support issue..."
              rows={1}
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="bg-indigo-600 text-white rounded-xl px-4 py-2.5 text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              {isStreaming ? '...' : 'Ask'}
            </button>
          </div>
          <div className="max-w-3xl mx-auto mt-1.5 text-xs text-gray-400 text-center">
            Internal tool · Answers based on resolved Zendesk tickets, Jira, Confluence & Slack
          </div>
        </div>
      </div>
    </div>
  )
}
