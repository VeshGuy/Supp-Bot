import { useState } from 'react'

const SESSION_KEY = 'supportiq_session_id'
const SESSIONS_LIST_KEY = 'supportiq_sessions'
export const MESSAGES_KEY = (id: string) => `supportiq_messages_${id}`

export interface SessionMeta {
  id: string
  title: string
  timestamp: number
}

function loadSessions(): SessionMeta[] {
  try {
    const stored = localStorage.getItem(SESSIONS_LIST_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

function saveSessions(sessions: SessionMeta[]) {
  localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(sessions))
}

export function useSession() {
  const [sessionId, setSessionId] = useState<string>(() => {
    const stored = localStorage.getItem(SESSION_KEY)
    if (stored) return stored
    const newId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, newId)
    return newId
  })

  const [sessions, setSessions] = useState<SessionMeta[]>(loadSessions)

  const updateSessionTitle = (id: string, firstMessage: string) => {
    const title = firstMessage.slice(0, 45) + (firstMessage.length > 45 ? '…' : '')
    setSessions(prev => {
      const exists = prev.find(s => s.id === id)
      const updated = exists
        ? prev.map(s => s.id === id ? { ...s, title } : s)
        : [{ id, title, timestamp: Date.now() }, ...prev]
      saveSessions(updated)
      return updated
    })
  }

  const resetSession = () => {
    const newId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, newId)
    setSessionId(newId)
  }

  const switchSession = (id: string) => {
    localStorage.setItem(SESSION_KEY, id)
    setSessionId(id)
  }

  const deleteSession = (id: string) => {
    localStorage.removeItem(MESSAGES_KEY(id))
    setSessions(prev => {
      const updated = prev.filter(s => s.id !== id)
      saveSessions(updated)
      return updated
    })
    if (id === sessionId) resetSession()
  }

  return { sessionId, sessions, updateSessionTitle, resetSession, switchSession, deleteSession }
}
