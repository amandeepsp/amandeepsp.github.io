const path = require("path")

const crateBlogPages = async (createPage, graphql) => {
    const postTemplate = path.resolve(`src/templates/blog-post.js`)
    const result = await graphql(`
        {
            allMarkdownRemark(
                sort: { order: DESC, fields: [frontmatter___date] }
                limit: 1000
            ) {
                edges {
                    node {
                        frontmatter {
                            path
                            layout
                            title
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

    const posts = result.data.allMarkdownRemark.edges.filter(({ node }) => (node.frontmatter.layout === 'blog-post'))

    posts.forEach(({ node }, index) => {
        const prev = index === 0 ? null : posts[index - 1].node
        const next = index === posts.length - 1 ? null : posts[index + 1].node
        createPage({
            path: node.frontmatter.path,
            component: postTemplate,
            context: {
                prev, next
            },
        })
    })
}

const createAboutPage = async (createPage, graphql) => {
    const aboutTemplate = path.resolve(`src/templates/about.js`)

    const result = await graphql(`
    {
        allMarkdownRemark(filter: {frontmatter: {layout: {eq: "about"}}}) {
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

    const { node:pageNode } = result.data.allMarkdownRemark.edges[0]

    createPage({
        path: pageNode.frontmatter.path,
        component: aboutTemplate,
    })


}

exports.createPages = async ({ actions, graphql }) => {
    const { createPage } = actions
    await crateBlogPages(createPage, graphql)
    await createAboutPage(createPage, graphql)
}
