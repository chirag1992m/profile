import React from 'react'

import { ListDetailView } from '../../src/components/ListDetail/ListDetailView'
import { WritingDetailView } from '../../src/components/WritingListDetail/WritingDetailView'
import { WritingList } from '../../src/components/WritingListDetail/WritingList'
import {
    getAllChapterMetadata,
    getCategorizedChapters,
    getChapter,
} from '../chapters'

export function generateStaticParams() {
    const posts = getAllChapterMetadata()

    const paths = posts.map((post) => ({
        slug: post.slug,
    }))

    return paths
}

export default function Page({
    params,
}: {
    params: { slug: string }
}): React.ReactElement {
    const post = getChapter(params.slug)

    return (
        <ListDetailView
            list={
                <WritingList
                    title="Machine Learning Book"
                    topNav="ml_book"
                    categorizedPosts={getCategorizedChapters()}
                ></WritingList>
            }
            hasDetail={Boolean(post)}
            detail={
                post && (
                    <WritingDetailView
                        topNav="ml_book"
                        postMetadata={post.postMetadata}
                        postContent={post.postContent}
                    ></WritingDetailView>
                )
            }
        />
    )
}
