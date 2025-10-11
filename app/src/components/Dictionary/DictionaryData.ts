// Dictionary data structure for word definitions
export interface DictionaryEntry {
    word: string
    definition: string
    pronunciation?: string
    etymology?: string
    examples?: string[]
}

// Dictionary database - you can expand this with more words
export const dictionary: Record<string, DictionaryEntry> = {
    transformer: {
        word: 'transformer',
        definition:
            'A deep learning model architecture introduced in "Attention Is All You Need" (2017) that uses self-attention mechanisms to process sequential data.',
        pronunciation: '/trænsˈfɔːrmər/',
        etymology:
            'From "transform" + "-er", referring to its ability to transform input sequences.',
        examples: [
            'The transformer architecture revolutionized natural language processing.',
            'GPT and BERT are both based on transformer models.',
        ],
    },
    agi: {
        word: 'AGI',
        definition:
            'Artificial General Intelligence - AI that can understand, learn, and apply knowledge across a wide range of tasks at a human level or beyond.',
        pronunciation: '/ˈeɪdʒiːaɪ/',
        examples: [
            'AGI would be capable of reasoning, problem-solving, and creativity across all domains.',
            'Unlike narrow AI, AGI would not be limited to specific tasks.',
        ],
    },
    llm: {
        word: 'LLM',
        definition:
            'Large Language Model - A type of AI model trained on vast amounts of text data to understand and generate human-like text.',
        pronunciation: '/ˈelˈelˈem/',
        examples: [
            'ChatGPT is an example of a large language model.',
            'LLMs can perform various text-based tasks like translation and summarization.',
        ],
    },
    adsense: {
        word: 'adsense',
        definition:
            "Google's advertising platform that allows website owners to display targeted ads and earn revenue.",
        pronunciation: '/ˈædsens/',
        examples: [
            'Many websites rely on AdSense for monetization.',
            'AdSense uses contextual targeting to show relevant ads.',
        ],
    },
    slop: {
        word: 'slop',
        definition:
            'In AI context, refers to low-quality, automated, or AI-generated content that lacks human oversight and quality.',
        pronunciation: '/slɒp/',
        examples: [
            'The internet is increasingly filled with AI slop.',
            'Content farms produce slop to game search algorithms.',
        ],
    },
    agentic: {
        word: 'agentic',
        definition:
            'Relating to AI agents that can act autonomously and make decisions without human intervention.',
        pronunciation: '/eɪˈdʒentɪk/',
        examples: [
            'Agentic AI can perform complex tasks independently.',
            'The future of AI lies in more agentic systems.',
        ],
    },
    bigtech: {
        word: 'BigTech',
        definition:
            'The largest and most dominant technology companies, typically referring to major corporations like Google, Apple, Microsoft, Amazon, Meta (Facebook), and others that have significant influence over technology markets and digital infrastructure.',
        pronunciation: '/ˈbɪɡtek/',
        etymology:
            'From "big" + "tech" (technology), referring to the largest technology companies.',
        examples: [
            'BigTech companies have enormous resources for AI research and development.',
            'Regulators are increasingly scrutinizing BigTech for antitrust concerns.',
            'BigTech firms like Google and Microsoft are leading the AI revolution.',
        ],
    },
    chaar_log_kya_kahenge: {
        word: 'chaar log kya kahenge',
        definition:
            'A hindi phrase used to describe the fear of what others (generally the close neighbors and family members) will say about you about your actions.',
        pronunciation: '/ˈtʃɑːr ˈlɒg ˈkja ˈkæˈhenʤi/',
    },
    social_media: {
        word: 'Social Media',
        definition:
            'Digital platforms and technologies that enable users to create, share, and interact with content and other users, often through personalized feeds and algorithms.',
        pronunciation: '/ˈsoʊʃəl ˈmidiə/',
        etymology:
            'From "social" (relating to society) + "media" (communication channels), referring to digital platforms for social interaction.',
        examples: [
            'Social media has transformed how people connect and share information.',
            'Platforms like Facebook, Instagram, TikTok, LinkedIn, etc. are examples of social media.',
            'Social media algorithms create personalized content feeds for users.',
        ],
    },
    ikigai: {
        word: 'ikigai',
        definition:
            'A Japanese concept meaning "reason for being" or "purpose in life" - the intersection of what you love, what you are good at, what the world needs, and what you can be paid for.',
        pronunciation: '/ˌiːkiˈɡaɪ/',
        etymology:
            'From Japanese 生き甲斐 (iki-gai), where iki means "life" and gai means "worth" or "value".',
        examples: [
            'Finding your ikigai can lead to a more fulfilling and purposeful life.',
            'The ikigai framework helps identify the sweet spot between passion, mission, profession, and vocation.',
            'Many people in Okinawa, Japan, credit their longevity to living according to their ikigai.',
        ],
    },
    iit: {
        word: 'IIT',
        definition:
            'Indian Institute of Technology - A group of autonomous public technical universities in India, widely regarded as the most prestigious engineering institutions in the country.',
        pronunciation: '/ˌaɪaɪˈtiː/',
        etymology:
            'Acronym for "Indian Institute of Technology", established to provide world-class technical education in India.',
        examples: [
            'IIT graduates are highly sought after by top companies worldwide.',
            'Getting into an IIT is considered a major achievement in India due to the extremely competitive entrance exam.',
            'The IIT system has produced many successful entrepreneurs and engineers globally.',
        ],
    },
}

// Helper function to check if a word exists in the dictionary
export const hasDefinition = (word: string): boolean => {
    return word.toLowerCase() in dictionary
}

// Helper function to get definition for a word
export const getDefinition = (word: string): DictionaryEntry | null => {
    return dictionary[word.toLowerCase()] ?? null
}
