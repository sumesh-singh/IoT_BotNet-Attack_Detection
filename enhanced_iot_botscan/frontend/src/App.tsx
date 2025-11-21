import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Provider } from 'react-redux'
import { store } from './app/store'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import AppShell from './layout/AppShell'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Training from './pages/Training'
import Models from './pages/Models'
import Evaluation from './pages/Evaluation'
import Drift from './pages/Drift'
import Logs from './pages/Logs'
import Settings from './pages/Settings'

const queryClient = new QueryClient()

export default function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <AppShell>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/training" element={<Training />} />
              <Route path="/models" element={<Models />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/drift" element={<Drift />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </QueryClientProvider>
    </Provider>
  )
}
