import React from "react"
import { graphql, Link } from "gatsby"
import Layout from "../components/layout"
import styled from "styled-components"
import SEO from "../components/seo"
import { DiscussionEmbed } from "disqus-react"

const BottomNavContainer = styled.div`
    display: flex;
    flex-direction: column;
    padding: 1rem 0;
`
const NextLink = styled.div`
    align-self: flex-end;
`

const SubTitle = styled.div`
    color: gray;
    padding: 1rem 0;
    font-size: 14px;
    font-weight: 600;
`

export default function Template({ data, pageContext }) {
    const { markdownRemark: post } = data
    const { next, prev } = pageContext;

    const disqusConfig = {
        shortname: process.env.DISQUS_SHORTNAME
    }

    return (
        <Layout>
            <SEO title={post.frontmatter.title} />
            <div>
                <h1>{post.frontmatter.title}</h1>
                <SubTitle>
                    Posted on {post.frontmatter.date} &bull; {post.timeToRead} min read
                </SubTitle>
                <div dangerouslySetInnerHTML={{ __html: post.html }} />
                <BottomNavContainer>
                    <h3>Read More</h3>
                    {prev &&
                        <Link className="link prev" to={prev.frontmatter.path}>
                            {`\u2bc7 ${prev.frontmatter.title}`}
                        </Link>}
                    {next &&
                        <NextLink>
                            <Link className="link next" to={next.frontmatter.path}>
                                {`${next.frontmatter.title} \u2bc8`}
                            </Link>
                        </NextLink>}
                </BottomNavContainer>
                <DiscussionEmbed {...disqusConfig} />
            </div>
        </Layout>
    )
}

export const pageQuery = graphql`
    query BlogPostByPath($path: String!) {
        markdownRemark(frontmatter: { path: { eq: $path } }) {
            html
            frontmatter {
                date(formatString: "MMMM DD, YYYY")
                path
                title
                description
            }
            timeToRead
        }
    }
`
