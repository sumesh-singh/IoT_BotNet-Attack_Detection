import { useDatasets, useValidateDataset } from '../api/datasets'

export default function Datasets() {
  const { data, isLoading, error } = useDatasets()
  const validate = useValidateDataset()
  const items = data?.items || []
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Datasets</h1>
      <div className="bg-white p-4 rounded shadow">
        {isLoading && <div>Loading...</div>}
        {error && <div className="text-red-600">Failed to load datasets</div>}
        {!isLoading && items.length === 0 && <div>No datasets</div>}
        {items.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="py-2">Name</th>
                <th className="py-2">Samples</th>
                <th className="py-2">Features</th>
                <th className="py-2">Classes</th>
                <th className="py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((d: any) => (
                <tr key={d.name} className="border-b">
                  <td className="py-2">{d.name}</td>
                  <td className="py-2">{d.samples ?? '-'}</td>
                  <td className="py-2">{d.features ?? '-'}</td>
                  <td className="py-2">{d.classes ?? '-'}</td>
                  <td className="py-2">
                    <button
                      className="px-3 py-1 bg-blue-600 text-white rounded"
                      onClick={() => validate.mutate(d.name)}
                      disabled={validate.isLoading}
                    >Validate</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {validate.isSuccess && (
        <div className="bg-green-100 border border-green-300 p-3 rounded">
          Validation complete
        </div>
      )}
      {validate.isError && (
        <div className="text-red-600">Validation failed</div>
      )}
    </div>
  )
}
