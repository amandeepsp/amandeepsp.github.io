import React from "react"
import styled from "styled-components"
import { Link } from "gatsby"

const CaptionSubdued = styled.h5`
    color: gray;
`

const TupleContainer = styled.div`
    width: 100%;
    display: flex;
    flex-direction: row;
    margin: 1rem 0;
    @media (max-width: 640px) {
        flex-direction: column;
    }
`

export default function PostTuple({
    post: {
        frontmatter: { path, title, date },
        excerpt,
    },
}) {
    return (
        <TupleContainer>
            <div>
                <h2>
                    <Link to={path}>{title}</Link>
                </h2>
                <CaptionSubdued>Posted on {date}</CaptionSubdued>
                <p>{excerpt}</p>
            </div>
        </TupleContainer>
    )
}
