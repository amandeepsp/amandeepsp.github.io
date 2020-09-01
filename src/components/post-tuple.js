import React from "react"
import styled from "styled-components"
import { Link } from "gatsby"

const CaptionSubdued = styled.h5`
    color: gray;
`

export default function PostTuple(props) {
    const { path, title, date } = props.post.frontmatter
    return (
        <div>
            <h2>
                <Link to={path}>{title}</Link>
            </h2>
            <CaptionSubdued>Posted on {date}</CaptionSubdued>
            <p>{props.post.excerpt}</p>
        </div>
    )
}
