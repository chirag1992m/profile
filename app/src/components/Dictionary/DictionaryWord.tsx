'use client'
import React from 'react';
import { DictionaryTooltip } from './DictionaryTooltip';
import { getDefinition } from './DictionaryData';

interface DictionaryWordProps {
    word: string;
    children?: React.ReactNode;
}

export const DictionaryWord: React.FC<DictionaryWordProps> = ({
    word,
    children
}) => {
    const definition = getDefinition(word);
    
    if (!definition) {
        // If no definition found, just return the word as-is
        return <span>{children || word}</span>;
    }

    return (
        <DictionaryTooltip word={word} definition={definition}>
            {children || word}
        </DictionaryTooltip>
    );
};
