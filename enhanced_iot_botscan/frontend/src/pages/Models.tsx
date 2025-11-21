import { useModels } from '../api/artifacts'

export default function Models() {
  const { data, isLoading } = useModels()
  const items = data?.items || []
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Models</h1>
      <div className="bg-white p-4 rounded shadow">
        {isLoading && <div>Loading...</div>}
        {!isLoading && items.length === 0 && <div>No models</div>}
        {items.length > 0 && (
          <ul className="list-disc pl-5">
            {items.map((m: any) => (
              <li key={m.name}>{m.name}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
