import Image from 'next/image'
import * as React from 'react'

import { type postMetadata } from '../../../writing/posts'
import { ListItem } from '../ListDetail/ListItem'

interface WritingListItemProps {
    post: postMetadata
    active: boolean
}

export const WritingListItem = React.memo<WritingListItemProps>(
    ({ post, active }) => {
        const aiIndicator = post.ai_assisted ? (
            <div className="flex items-center justify-center w-4 h-4">
                <Image
                    src="/static/humanoid_icon.png"
                    alt="AI Assisted"
                    width={16}
                    height={16}
                    className="opacity-60 hover:opacity-100 transition-opacity"
                    title="This post was written with AI assistance"
                />
            </div>
        ) : (
            <div className="flex items-center justify-center w-4 h-4">
                <Image
                    src="/static/all_human_icon.png"
                    alt="Human Written"
                    width={16}
                    height={16}
                    className="opacity-60 hover:opacity-100 transition-opacity"
                    title="This post was written entirely by a human"
                />
            </div>
        )

        return (
            <ListItem
                key={`wli_${post.slug}`}
                href="/writing/[slug]"
                as={`/writing/${post.slug}`}
                title={post.title}
                description={post.subtitle}
                byline={`${post.date}`}
                active={active}
                leadingAccessory={aiIndicator}
            />
        )
    }
)
