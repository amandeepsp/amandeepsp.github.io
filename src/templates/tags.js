import React from "react"
import PostTuple from "../components/post-tuple"
import Layout from "../components/layout"
import { Link, graphql } from "gatsby"

const Tags = ({ pageContext, data }) => {
    const { tag } = pageContext
    const { nodes, totalCount } = data.allMarkdownRemark
    const tagHeader = `${totalCount} post${totalCount === 1 ? "" : "s"} tagged with "${tag}"`

    const blogList = nodes.map((post) => {
        return <PostTuple post={post} key={post.id} />
    })

    return (
        <Layout>
            <h2>{tagHeader}</h2>
            {blogList}
            <Link to="/tags">
                <h4>All tags</h4>
            </Link>
        </Layout>
    )
}

export default Tags

export const pageQuery = graphql`
    query tagQuery($tag: String) {
        allMarkdownRemark(
            limit: 2000
            sort: { fields: [frontmatter___date], order: DESC }
            filter: { frontmatter: { categories: { in: [$tag] } } }
        ) {
            totalCount
            nodes {
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
`
