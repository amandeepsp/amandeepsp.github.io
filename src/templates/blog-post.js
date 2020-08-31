import React from "react"
import { graphql, Link } from "gatsby"
import Layout from "../components/layout"
import styled from "styled-components"
import SEO from "../components/seo"
import { DiscussionEmbed } from "disqus-react"
import { DISQUS_SHORTNAME } from "../utils/constants"
import kebabCase from "lodash/kebabCase"

const BottomNavContainer = styled.div`
    display: flex;
    flex-direction: row;
    padding: 1rem 0;
`
const BottomLink = styled.div`
    padding: 0 0.5rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    flex-basis:0;
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
        shortname: DISQUS_SHORTNAME
    }

    return (
        <Layout>
            <SEO title={post.frontmatter.title} />
            <div>
                <h1>{post.frontmatter.title}</h1>
                <SubTitle>
                    Posted on {post.frontmatter.date} &bull; {post.timeToRead} min read 
                    &bull; Tagged with {post.frontmatter.categories.map(tag => {
                        return(
                            <Link style={{ padding: "0.2rem"}}to={`/tags/${kebabCase(tag)}/`}>{tag}</Link>
                        )
                    })}
                </SubTitle>
                <div dangerouslySetInnerHTML={{ __html: post.html }} />
                <BottomNavContainer>
                    {prev &&
                        <BottomLink>
                            <h6>Previous</h6>
                            <Link to={prev.frontmatter.path}>&larr; {prev.frontmatter.title}</Link>
                        </BottomLink>
                    }
                    {next &&
                        <BottomLink style={{
                            alignItems: 'flex-end'
                        }}>
                            <h6>Next</h6>
                            <Link to={next.frontmatter.path}>{next.frontmatter.title} &rarr;</Link>
                        </BottomLink>
                    }
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
                categories
            }
            timeToRead
        }
    }
`
