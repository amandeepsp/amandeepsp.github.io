import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import TagTuple from "../components/tag-tuple"
import styled from "styled-components"

const TagsDiv = styled.div`
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
`

const TagsPage = ({
    data: {
        allMarkdownRemark: { group },
    },
}) => (
    <Layout>
        <h2>Tags</h2>
        <TagsDiv>
            {group.map((tag) => (
                <TagTuple key={tag.fieldValue} tag={tag} />
            ))}
        </TagsDiv>
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
