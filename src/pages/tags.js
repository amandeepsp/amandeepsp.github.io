import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import kebabCase from "lodash/kebabCase"
import { Link } from "gatsby"
import { SecondaryHeader } from "../components/styled"

const TagsPage = ({
    data: {
        allMarkdownRemark: { group },
    },
}) => (
    <Layout>
        <SecondaryHeader>Tags</SecondaryHeader>
        <ul style={{ marginTop: "1em" }}>
            {group.map((tag) => (
                <li key={tag.fieldValue}>
                    <Link key={tag.fieldValue} to={`/tags/${kebabCase(tag.fieldValue)}/`}>
                        {tag.fieldValue} ({tag.totalCount})
                    </Link>
                </li>
            ))}
        </ul>
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
