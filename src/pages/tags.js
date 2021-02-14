import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import kebabCase from "lodash/kebabCase"
import { Link } from "gatsby"
import { SecondaryHeader } from "../components/styled"
import styled from "styled-components"

const TagsContainer = styled.div`
    margin-top: 1em;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    align-items: baseline;
`

const TagsPage = ({
    data: {
        allMarkdownRemark: { group },
    },
}) => (
        <Layout>
            <SecondaryHeader>Tags</SecondaryHeader>
            <TagsContainer>
                {group.map((tag) => (
                    <Link style={{ fontSize: `${tag.totalCount}em`, marginRight: "0.5em" }}
                        key={tag.fieldValue}
                        to={`/tags/${kebabCase(tag.fieldValue)}/`}
                    >
                        {tag.fieldValue} ({tag.totalCount})
                    </Link>
                ))}
            </TagsContainer>
        </Layout>
    )

export default TagsPage

export const pageQuery = graphql`
    query allTagsQuery {
        allMarkdownRemark(limit: 2000) {
            group(field: frontmatter___categories) {
                fieldValue
                totalCount
            }
        }
    }
`
