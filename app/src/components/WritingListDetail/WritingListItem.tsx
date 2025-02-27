import * as React from 'react'

import { type postMetadata } from '../../../writing/posts'
import { ListItem } from '../ListDetail/ListItem'

interface WritingListItemProps {
    post: postMetadata
    topNav: string
    active: boolean
}

export const WritingListItem = React.memo<WritingListItemProps>(
    ({ post, topNav, active }) => {
        return (
            <ListItem
                key={`wli_${post.slug}`}
                href={`/${topNav}/[slug]`}
                as={`/${topNav}/${post.slug}`}
                title={post.title}
                description={post.subtitle}
                byline={`${post.date}`}
                active={active}
            />
        )
    }
)
