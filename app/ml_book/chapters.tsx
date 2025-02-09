import fs from 'fs'
import matter from 'gray-matter'
import { type WritingDetailProps } from '../src/components/WritingListDetail/WritingDetailView'
import { type postMetadata } from '../writing/posts'

const MLBookFolder = 'public/ml_book_chapters'

export interface chapterMetadata {
    slug: string
    index: number
    chapter: string
    title: string
    sub_index: number
    subtitle: string
    date: string
    cover_image: string
    cover_image_prompt?: string
}

export const getAllChapterMetadata = (): chapterMetadata[] => {
    const inDevEnvironment = !!process && process.env.NODE_ENV === 'development'

    const markdownChapter = fs
        .readdirSync(MLBookFolder)
        .filter((file) => file.endsWith('.md'))
        .filter((file) =>
            inDevEnvironment ? true : !file.startsWith('ignore-')
        )

    const chaptersMetadata = markdownChapter.map((filename) => {
        const filePath = `${MLBookFolder}/${filename}`
        const fileContent = fs.readFileSync(filePath, 'utf-8')
        const matterResult = matter(fileContent)

        return {
            slug: matterResult.data.slug,
            index: matterResult.data.index,
            chapter: matterResult.data.chapter,
            title: matterResult.data.title,
            sub_index: matterResult.data.sub_index,
            subtitle: matterResult.data.subtitle,
            date: matterResult.data.date,
            cover_image: matterResult.data.cover_image,
            cover_image_prompt: matterResult.data.cover_image_prompt
                ? matterResult.data.cover_image_prompt
                : '',
        }
    })

    return chaptersMetadata
}

export const getCategorizedChapters = (): Record<string, postMetadata[]> => {
    const chaptersMetadata = getAllChapterMetadata().sort((a, b) => {
        // Replace with the correct numeric comparison if index/sub_index are strings
        return a.index - b.index || a.sub_index - b.sub_index
    })

    const categories: Record<string, postMetadata[]> = chaptersMetadata.reduce(
        (x, y) => {
            ;(x[y.chapter] = x[y.chapter] || []).push({
                slug: y.slug,
                title: y.title,
                subtitle: y.subtitle,
                category: y.chapter,
                date: y.date,
                cover_image: y.cover_image,
                cover_image_prompt: y.cover_image_prompt ? y.cover_image_prompt : '',
            })
            return x
        },
        {}
    )

    return categories
}

export const getChapter = (slug: string): WritingDetailProps | null => {
    const markdownSlugs = getAllChapterMetadata().map((chapterMetadata) => {
        return chapterMetadata.slug
    })

    if (markdownSlugs.includes(slug)) {
        const filePath = `${MLBookFolder}/${slug}.md`
        const fileContent = fs.readFileSync(filePath, 'utf-8')
        const matterResult = matter(fileContent)

        return {
            topNav: 'ml_book',
            postMetadata: {
                slug: matterResult.data.slug,
                title: matterResult.data.title,
                subtitle: matterResult.data.subtitle,
                category: matterResult.data.chapter,
                date: matterResult.data.date,
                cover_image: matterResult.data.cover_image,
                cover_image_prompt: matterResult.data.cover_image_prompt
                    ? matterResult.data.cover_image_prompt
                    : '',
            },
            postContent: matterResult.content,
        }
    } else {
        return null
    }
}
