import React from "react"

import kebabCase from "lodash/kebabCase"

import { Link, graphql } from "gatsby"
import Layout from "../components/layout"

const TagsPage = ({
    data: {
        allMarkdownRemark: { group },
    },
}) => (
    <Layout>
        <h2>Tags</h2>
        <div style={{display: "flex", flexDirection: "column"}}>
        {group.map((tag) => (
            <Link
                key={tag.fieldValue}
                to={`/tags/${kebabCase(tag.fieldValue)}/`}
            >
                {tag.fieldValue} ({tag.totalCount})
            </Link>
        ))}
        </div>
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
