'use client'
import React, { useState, useRef, useEffect } from 'react';
import { DictionaryEntry } from './DictionaryData';

interface DictionaryTooltipProps {
    word: string;
    definition: DictionaryEntry;
    children: React.ReactNode;
}

export const DictionaryTooltip: React.FC<DictionaryTooltipProps> = ({
    word,
    definition,
    children
}) => {
    const [isVisible, setIsVisible] = useState(false);
    const [position, setPosition] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLSpanElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);

    const updatePosition = () => {
        if (triggerRef.current && tooltipRef.current) {
            const triggerRect = triggerRef.current.getBoundingClientRect();
            const tooltipRect = tooltipRef.current.getBoundingClientRect();
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;

            let top = triggerRect.bottom + 8;
            let left = triggerRect.left;

            // Adjust horizontal position if tooltip would overflow
            if (left + tooltipRect.width > viewportWidth - 16) {
                left = viewportWidth - tooltipRect.width - 16;
            }
            if (left < 16) {
                left = 16;
            }

            // Adjust vertical position if tooltip would overflow
            if (top + tooltipRect.height > viewportHeight - 16) {
                top = triggerRect.top - tooltipRect.height - 8;
            }

            setPosition({ top, left });
        }
    };

    useEffect(() => {
        if (isVisible) {
            updatePosition();
        }
    }, [isVisible]);

    const handleMouseEnter = () => {
        setIsVisible(true);
    };

    const handleMouseLeave = () => {
        setIsVisible(false);
    };

    return (
        <>
            <span
                ref={triggerRef}
                className="dictionary-word cursor-help border-b border-dotted border-blue-400 text-blue-600 hover:text-blue-800 transition-colors"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </span>
            
            {isVisible && (
                <div
                    ref={tooltipRef}
                    className="dictionary-tooltip fixed z-50 max-w-sm p-4 bg-white border border-gray-200 rounded-lg shadow-lg"
                    style={{
                        top: `${position.top}px`,
                        left: `${position.left}px`,
                    }}
                    onMouseEnter={handleMouseEnter}
                    onMouseLeave={handleMouseLeave}
                >
                    <div className="space-y-2">
                        <div className="flex items-center gap-2">
                            <h3 className="font-semibold text-gray-900 text-lg">
                                {definition.word}
                            </h3>
                            {definition.pronunciation && (
                                <span className="text-sm text-gray-500 italic">
                                    {definition.pronunciation}
                                </span>
                            )}
                        </div>
                        
                        <p className="text-gray-700 text-sm leading-relaxed">
                            {definition.definition}
                        </p>
                        
                        {definition.etymology && (
                            <div className="text-xs text-gray-500">
                                <span className="font-medium">Origin:</span> {definition.etymology}
                            </div>
                        )}
                        
                        {definition.examples && definition.examples.length > 0 && (
                            <div className="text-xs text-gray-600">
                                <span className="font-medium">Examples:</span>
                                <ul className="mt-1 space-y-1">
                                    {definition.examples.map((example, index) => (
                                        <li key={index} className="italic">
                                            "{example}"
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </>
    );
};
