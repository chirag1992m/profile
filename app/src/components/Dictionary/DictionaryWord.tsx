'use client'
import React from 'react'
import { getDefinition } from './DictionaryData'
import { DictionaryTooltip } from './DictionaryTooltip'

interface DictionaryWordProps {
    word: string
    children?: React.ReactNode
}

export const DictionaryWord: React.FC<DictionaryWordProps> = ({
    word,
    children,
}): React.ReactElement => {
    const definition = getDefinition(word)

    if (definition === null) {
        // If no definition found, just return the word as-is
        return <span>{children ?? word}</span>
    }

    return (
        <DictionaryTooltip word={word} definition={definition}>
            {children ?? word}
        </DictionaryTooltip>
    )
}
