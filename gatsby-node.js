const path = require("path")
const _ = require("lodash")

const crateBlogPages = async (actions, graphql) => {
    const { createPage, createRedirect, reporter } = actions

    const postTemplate = path.resolve(`src/templates/blog-post.js`)
    const result = await graphql(`
        {
            allMarkdownRemark(
                sort: { order: DESC, fields: [frontmatter___date] }
                limit: 1000
                filter: { frontmatter: { layout: { eq: "blog-post" } } }
            ) {
                edges {
                    node {
                        frontmatter {
                            path
                            layout
                            title
                            categories
                            redirects
                        }
                    }
                }
            }
        }
    `)

    if (result.errors) {
        reporter.panicOnBuild(`Error while running GraphQL query.`)
        return
    }

    const posts = result.data.allMarkdownRemark.edges.filter(
        ({ node }) => node.frontmatter.layout === "blog-post"
    )

    posts.forEach(({ node }, index) => {
        const prev = index === 0 ? null : posts[index - 1].node
        const next = index === posts.length - 1 ? null : posts[index + 1].node

        const {
            frontmatter: { path, redirects },
        } = node

        if (redirects) {
            redirects.forEach((fromPath) => {
                createRedirect({
                    fromPath,
                    toPath: path,
                    redirectInBrowser: true,
                    isPermanent: true,
                })
            })
        }

        createPage({
            path: path,
            component: postTemplate,
            context: {
                prev,
                next,
            },
        })
    })

    const tagGroups = await graphql(`
        {
            allMarkdownRemark(
                limit: 2000
                filter: { frontmatter: { layout: { eq: "blog-post" } } }
            ) {
                group(field: frontmatter___categories) {
                    fieldValue
                }
            }
        }
    `)

    if (tagGroups.errors) {
        reporter.panicOnBuild(`Error while running GraphQL query.`)
        return
    }

    const tags = tagGroups.data.allMarkdownRemark.group
    const tagTemplate = path.resolve(`src/templates/tags.js`)

    tags.forEach((tag) => {
        createPage({
            path: `/tags/${_.kebabCase(tag.fieldValue)}/`,
            component: tagTemplate,
            context: {
                tag: tag.fieldValue,
            },
        })
    })
}

const createAboutPage = async (actions, graphql) => {
    const { createPage, reporter } = actions

    const aboutTemplate = path.resolve(`src/templates/about.js`)

    const result = await graphql(`
        {
            allMarkdownRemark(
                filter: { frontmatter: { layout: { eq: "about" } } }
            ) {
                edges {
                    node {
                        frontmatter {
                            path
                            title
                            layout
                        }
                    }
                }
            }
        }
    `)

    if (result.errors) {
        reporter.panicOnBuild(`Error while running GraphQL query.`)
        return
    }

    const { node: pageNode } = result.data.allMarkdownRemark.edges[0]

    createPage({
        path: pageNode.frontmatter.path,
        component: aboutTemplate,
    })
}

exports.createPages = async ({ actions, graphql }) => {
    await crateBlogPages(actions, graphql)
    await createAboutPage(actions, graphql)
}

exports.onCreateWebpackConfig = ({ stage, loaders, actions }) => {
    if (stage === "build-html") {
        actions.setWebpackConfig({
            module: {
                rules: [
                    {
                        test: [/node_modules\/p5/, /node_modules\\p5/],
                        use: loaders.null(),
                    },
                ],
            },
        })
    }
}
