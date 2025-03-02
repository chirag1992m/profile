import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter'
import PlotlyBlock from './PlotlyBlock'

interface CodeBlockProps {
  text: string
  language: string
  [key: string]: unknown
}

export function CodeBlock({
  text,
  language,
  ...rest
}: CodeBlockProps): React.ReactElement {
  if (language === 'plotly') {
    return <PlotlyBlock src={text} />
  } else {
    return (
      <SyntaxHighlighter
        showLineNumbers={true}
        useInlineStyles={false}
        language={language}
        wrapLongLines
        {...rest}
      >
        {text}
      </SyntaxHighlighter>
    )
  }
}