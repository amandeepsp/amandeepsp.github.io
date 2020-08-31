import React from "react"

import Layout from "../components/layout"
import { Link, graphql } from "gatsby"

const Tags = ({ pageContext, data }) => {
    const { tag } = pageContext
    const { edges, totalCount } = data.allMarkdownRemark
    const tagHeader = `${totalCount} post${
        totalCount === 1 ? "" : "s"
        } tagged with "${tag}"`

    const blogList = edges.map(({ node }) => {
        const { title, path } = node.frontmatter
        return (
            <li key={path}>
                <Link to={path}>{title}</Link>
            </li>
        )
    })

    return (
        <Layout>
            <h2>{tagHeader}</h2>
            <ul>
                {blogList}
            </ul>
            <Link to="/tags"><h4>All tags</h4></Link>
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
      edges {
        node {
          frontmatter {
            title
            path
          }
        }
      }
    }
  }
`