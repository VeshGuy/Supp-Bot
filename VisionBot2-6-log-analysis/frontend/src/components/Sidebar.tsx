import type { SessionMeta } from '../hooks/useSession'

interface Props {
  sessions: SessionMeta[]
  currentSessionId: string
  onSelect: (id: string) => void
  onNewChat: () => void
  onDelete: (id: string) => void
}

function groupSessions(sessions: SessionMeta[]) {
  const now = Date.now()
  const today: SessionMeta[] = []
  const yesterday: SessionMeta[] = []
  const older: SessionMeta[] = []

  for (const s of sessions) {
    const age = now - s.timestamp
    if (age < 86400000) today.push(s)
    else if (age < 172800000) yesterday.push(s)
    else older.push(s)
  }
  return { today, yesterday, older }
}

function SessionGroup({ label, sessions, currentId, onSelect, onDelete }: {
  label: string
  sessions: SessionMeta[]
  currentId: string
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  if (!sessions.length) return null
  return (
    <div className="mb-3">
      <div className="text-xs text-gray-400 px-3 py-1 font-medium uppercase tracking-wide">{label}</div>
      {sessions.map(s => (
        <div
          key={s.id}
          className={`group flex items-center gap-1 px-3 py-2 rounded-lg mx-1 cursor-pointer text-sm transition-colors ${
            s.id === currentId
              ? 'bg-indigo-50 text-indigo-700 font-medium'
              : 'text-gray-700 hover:bg-gray-100'
          }`}
          onClick={() => onSelect(s.id)}
        >
          <span className="flex-1 truncate">{s.title || 'New conversation'}</span>
          <button
            onClick={e => { e.stopPropagation(); onDelete(s.id) }}
            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 text-xs px-1 transition-opacity shrink-0"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}

export function Sidebar({ sessions, currentSessionId, onSelect, onNewChat, onDelete }: Props) {
  const { today, yesterday, older } = groupSessions(
    [...sessions].sort((a, b) => b.timestamp - a.timestamp)
  )

  return (
    <div className="w-60 shrink-0 bg-white border-r border-gray-200 flex flex-col h-screen sticky top-0">
      {/* Header */}
      <div className="px-3 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-xs">S</div>
          <span className="font-semibold text-gray-900 text-sm">SupportIQ</span>
        </div>
        <button
          onClick={onNewChat}
          className="w-full text-sm bg-indigo-600 text-white rounded-lg py-2 hover:bg-indigo-700 transition-colors font-medium"
        >
          + New chat
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto py-2">
        {sessions.length === 0 ? (
          <div className="text-xs text-gray-400 text-center px-3 py-6">No previous chats</div>
        ) : (
          <>
            <SessionGroup label="Today" sessions={today} currentId={currentSessionId} onSelect={onSelect} onDelete={onDelete} />
            <SessionGroup label="Yesterday" sessions={yesterday} currentId={currentSessionId} onSelect={onSelect} onDelete={onDelete} />
            <SessionGroup label="Older" sessions={older} currentId={currentSessionId} onSelect={onSelect} onDelete={onDelete} />
          </>
        )}
      </div>

      <div className="px-3 py-3 border-t border-gray-100 text-xs text-gray-400 text-center">
        Internal · Voice API
      </div>
    </div>
  )
}
