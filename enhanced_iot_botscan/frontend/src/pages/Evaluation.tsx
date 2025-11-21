import { useReports } from '../api/artifacts'

export default function Evaluation() {
  const { data, isLoading } = useReports()
  const items = data?.items || []
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Evaluation</h1>
      <div className="bg-white p-4 rounded shadow">
        {isLoading && <div>Loading...</div>}
        {!isLoading && items.length === 0 && <div>No reports</div>}
        {items.length > 0 && (
          <ul className="list-disc pl-5">
            {items.map((r: any) => (
              <li key={r.name}>{r.name}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
