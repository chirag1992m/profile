---
slug: 'ignore-example-blog-post'
title: 'Example Blogpost'
subtitle: 'Example blog, showing markdown powers'
category: 'Ignore (only dev)'
date: '2024-05-15'
cover_image: '/blog_images/ignore-example-blog-post/plant_lifecycle.webp'
cover_image_prompt: 'Some prompt for the cover image'
---

Hello this is my first blog after a long time on XGBoost.

`react-markdown` is a markdown component for React.

👉 Changes are re-rendered as you type.

👈 Try writing some markdown on the left.

## Overview

-   Follows [CommonMark](https://commonmark.org)
-   Optionally follows [GitHub Flavored Markdown](https://github.github.com/gfm/)
-   Renders actual React elements instead of using `dangerouslySetInnerHTML`
-   Lets you define your own components (to render `MyHeading` instead of `'h1'`)
-   Has a lot of plugins

## Contents

Here is an example of a plugin in action
([`remark-toc`](https://github.com/remarkjs/remark-toc)).
**This section is replaced by an actual table of contents**.

### Collapsible Blocks

You can create collapsible with the `details` and `summary` HTML5 tags.

<details>
<summary>Collapsible Block Heading</summary>

#### Collapsible Block Hidden Heading

Content here

</details>

## Syntax highlighting

Here is an example of a plugin to highlight code:
[`rehype-highlight`](https://github.com/rehypejs/rehype-highlight).

```python
import React from 'react'
import ReactDOM from 'react-dom'
import Markdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'

const markdown = `
# Your markdown here
`

ReactDOM.render(
    <Markdown rehypePlugins={[rehypeHighlight]}>{markdown}</Markdown>,
    document.querySelector('#content')
)
```

Pretty neat, eh?

## GitHub flavored markdown (GFM)

For GFM, you can _also_ use a plugin:
[`remark-gfm`](https://github.com/remarkjs/react-markdown#use).
It adds support for GitHub-specific extensions to the language:
tables, strikethrough, tasklists, and literal URLs.

These features **do not work by default**.
👆 Use the toggle above to add the plugin.

|    Feature | Support              |
| ---------: | :------------------- |
| CommonMark | 100%                 |
|        GFM | 100% w/ `remark-gfm` |

~~strikethrough~~

-   [ ] task list
-   [x] checked item

https://example.com

## HTML in markdown

⚠️ HTML in markdown is quite unsafe, but if you want to support it, you can
use [`rehype-raw`](https://github.com/rehypejs/rehype-raw).
You should probably combine it with
[`rehype-sanitize`](https://github.com/rehypejs/rehype-sanitize).

<blockquote>
  👆 Use the toggle above to add the plugin.
</blockquote>

Or create a footnote[^1]

[^1]: The big note

## Components

You can pass components to change things:

```js
import React from 'react'
import ReactDOM from 'react-dom'
import Markdown from 'react-markdown'
import MyFancyRule from './components/my-fancy-rule.js'

const markdown = `
# Your markdown here
`

ReactDOM.render(
    <Markdown
        components={{
            // Use h2s instead of h1s
            h1: 'h2',
            // Use a component instead of hrs
            hr(props) {
                const { node, ...rest } = props
                return <MyFancyRule {...rest} />
            },
        }}
    >
        {markdown}
    </Markdown>,
    document.querySelector('#content')
)
```

## React Math

Here you can write equations like this:

$$
L = \frac{1}{2} \rho v^2 S C_L
$$

Or like this:

```math
L = \frac{1}{2} \rho v^2 S C_L
```

## React Graphs by Plotly

```plotly
/blog_images/ignore-example-blog-post/plot_1.json
```

## React Diagrams by PlantUML

Create your diagram using the [`PlantUML jar`](https://plantuml.com/download) and use that to generate the SVG representation of the diagram and embed as an image.

Command to generate the SVG image:

```bash
java -jar plantuml.jar -tsvg /path/to/text/file/containing/plantUML_schema.puml
```

![PlantUML SVG](/blog_images/ignore-example-blog-post/plantuml_example.svg)

---

## More info?

Much more info is available in the
[readme on GitHub](https://github.com/remarkjs/react-markdown)!
