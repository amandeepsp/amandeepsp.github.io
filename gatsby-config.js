module.exports = {
	siteMetadata: {
		title: "corpus curiosum",
		author: "Amandeep Singh",
		description: "Blog about my ramblings in software engineering, from the most pedantic to the highly esoteric",
		siteUrl: "https://amandeepsp.github.io/",
		social: [
			{
				website: "github",
				username: "amandeepsp"
			},
			{
				website: "linkedin",
				username: "amandeepsp"
			},
			{
				website: "twitter",
				username: "theamndeepsingh"
			},
		],
		headerNav: [
			{
				title: "Blog",
				url: "/blog"
			},
			{
				title: "About",
				url: "/about"
			}
		]
	},
	plugins: [
		"gatsby-plugin-catch-links",
		"gatsby-plugin-react-helmet",
		"gatsby-plugin-sharp",
		"gatsby-plugin-styled-components",
		"gatsby-plugin-twitter",
		"gatsby-plugin-sitemap",
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
					"gatsby-remark-images",
					"gatsby-remark-prismjs",
					"gatsby-remark-katex",
					"gatsby-remark-copy-linked-files",
					"gatsby-remark-embedder"
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
			resolve: `gatsby-plugin-google-analytics`,
			options: {
				trackingId: process.env.GA_TRACKING_ID
			}
		},
	],
}
