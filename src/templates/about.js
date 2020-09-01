import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import SEO from "../components/seo"

export default function About({ data }) {
    const { markdownRemark: post } = data

    return (
        <Layout>
            <SEO title={post.frontmatter.title} />
            <div>
                <h1>{post.frontmatter.title}</h1>
                <div dangerouslySetInnerHTML={{ __html: post.html }} />
            </div>
        </Layout>
    )
}

export const pageQuery = graphql`
    query AboutPageQuery {
        markdownRemark(frontmatter: { path: { eq: "/about" } }) {
            html
            frontmatter {
                date(formatString: "MMMM DD, YYYY")
                path
                title
            }
        }
    }
`
