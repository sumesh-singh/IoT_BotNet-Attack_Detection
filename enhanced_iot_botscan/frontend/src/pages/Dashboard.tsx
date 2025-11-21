export default function Dashboard() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded shadow">Training Runs</div>
        <div className="bg-white p-4 rounded shadow">Alerts</div>
        <div className="bg-white p-4 rounded shadow">Dataset Stats</div>
      </div>
    </div>
  )
}
