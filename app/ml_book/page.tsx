import React from 'react'

import { ListDetailView } from '../src/components/ListDetail/ListDetailView'
import { WritingList } from '../src/components/WritingListDetail/WritingList'
import { getCategorizedChapters } from './chapters'

export default function Page(): React.ReactElement {
    return (
        <ListDetailView
            list={
                <WritingList
                    title="Machine Learning Book"
                    topNav="ml_book"
                    categorizedPosts={getCategorizedChapters()}
                ></WritingList>
            }
            hasDetail={false}
            detail={null}
        />
    )
}
