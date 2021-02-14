/* eslint-disable no-undef */
module.exports = {
    siteMetadata: {
        title: "corpus curiosum",
        author: "Amandeep Singh",
        description:
            "Blog about my ramblings in software engineering, from the most pedantic to the highly esoteric",
        siteUrl: "https://amandeepsp.github.io/",
        social: [
            {
                website: "github",
                username: "amandeepsp",
            },
            {
                website: "linkedin",
                username: "amandeepsp",
            },
            {
                website: "twitter",
                username: "theamndeepsingh",
            },
        ],
        headerNav: [
            {
                title: "Tags",
                url: "/tags",
            },
            {
                title: "About",
                url: "/about",
            },
        ],
    },
    plugins: [
        "gatsby-plugin-catch-links",
        "gatsby-plugin-react-helmet",
        "gatsby-plugin-sharp",
        "gatsby-plugin-styled-components",
        "gatsby-plugin-sitemap",
        "gatsby-plugin-meta-redirect",
        "gatsby-plugin-catch-links",
        {
            resolve: "gatsby-source-filesystem",
            options: {
                path: `${__dirname}/content`,
                name: "content",
            },
        },
        {
            resolve: "gatsby-transformer-remark",
            options: {
                plugins: [
                    {
                        resolve: "gatsby-remark-images",
                        options: {
                            maxWidth: 540,
                            showCaptions: true,
                            markdownCaptions: true,
                        },
                    },
                    "gatsby-remark-katex",
                    "gatsby-remark-copy-linked-files",
                    {
                        resolve: "gatsby-remark-prismjs",
                        options: {
                            inlineCodeMarker: ">",
                        },
                    },
                ],
            },
        },
        {
            resolve: `gatsby-plugin-typography`,
            options: {
                pathToConfigModule: `src/utils/typography`,
            },
        },
        {
            resolve: `gatsby-plugin-gtag`,
            options: {
                trackingId: "UA-112449883-1",
                head: true,
                anonymize: true,
            },
        },
    ],
}
