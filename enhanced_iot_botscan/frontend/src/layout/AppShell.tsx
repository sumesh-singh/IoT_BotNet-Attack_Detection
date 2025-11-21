import { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

type Props = { children: ReactNode }

export default function AppShell({ children }: Props) {
  return (
    <div className="h-full flex">
      <aside className="w-64 bg-gray-900 text-white flex flex-col">
        <div className="px-4 py-3 text-lg font-semibold">IoT BotScan</div>
        <nav className="flex-1 px-2 space-y-1">
          <NavLink to="/dashboard" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Dashboard</NavLink>
          <NavLink to="/datasets" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Datasets</NavLink>
          <NavLink to="/training" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Training</NavLink>
          <NavLink to="/evaluation" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Evaluation</NavLink>
          <NavLink to="/models" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Models</NavLink>
          <NavLink to="/drift" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Drift</NavLink>
          <NavLink to="/logs" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Logs</NavLink>
          <NavLink to="/settings" className={({ isActive }) => `block px-3 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`}>Settings</NavLink>
        </nav>
      </aside>
      <main className="flex-1 bg-gray-100">
        <header className="h-14 bg-white border-b flex items-center px-4">Dashboard</header>
        <div className="p-4">{children}</div>
      </main>
    </div>
  )
}
