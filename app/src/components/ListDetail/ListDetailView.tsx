export interface ListDetailViewProps {
    list: React.ReactElement | null
    detail: React.ReactElement | null
    hasDetail?: boolean
}

export function ListDetailView({
    list,
    detail,
    hasDetail = false,
}: ListDetailViewProps): React.ReactElement {
    return (
        <div className="flex w-full">
            {list != null && (
                <div
                    id="list"
                    className={`bg-dots ${
                        hasDetail ? 'hidden xl:flex' : 'min-h-screen w-full'
                    }`}
                >
                    {list}
                </div>
            )}
            {detail}
        </div>
    )
}
