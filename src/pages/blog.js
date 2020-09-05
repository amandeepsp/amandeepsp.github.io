import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import SEO from "../components/seo"
import PostTuple from "../components/post-tuple"

export default function Blog({ data }) {
    const { edges: posts } = data.allMarkdownRemark

    const postsToRender = posts
        .filter((post) => post.node.frontmatter.title.length > 0)
        .map(({ node: post }) => <PostTuple post={post} key={post.id} />)

    return (
        <Layout>
            <SEO title={"Blog"} />
            <h2>All Posts</h2>
            {postsToRender}
        </Layout>
    )
}

export const pageQuery = graphql`
    query BlogQuery {
        allMarkdownRemark(
            sort: { order: DESC, fields: [frontmatter___date] }
            filter: { frontmatter: { layout: { eq: "blog-post" } } }
        ) {
            edges {
                node {
                    excerpt(pruneLength: 250)
                    id
                    frontmatter {
                        title
                        date(formatString: "MMMM DD, YYYY")
                        path
                    }
                }
            }
        }
    }
`
