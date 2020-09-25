import React from "react"
import styled from "styled-components"
import { Link } from "gatsby"
import { circlePacking, tiledLines, cubicDisarray } from "../art"
import { useP5 } from "../hooks/use-p5"
import _ from "lodash"

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

const TupleImage = styled.div`
    padding-right: 1rem;
    align-self: center;
    @media (max-width: 640px) {
        & > canvas {
            width: 100% !important;
            height: 100% !important;
        }
    }
`
const artTypes = {
    tiled_lines: tiledLines,
    circle_packing: circlePacking,
    cubic_disarray: cubicDisarray,
}

export default function PostTuple({
    post: {
        frontmatter: { path, title, date, art_type },
        excerpt,
    },
}) {
    const sketch = art_type
        ? artTypes[art_type]
        : _.sample([tiledLines, circlePacking, cubicDisarray])
    const [setRef] = useP5(sketch)
    return (
        <TupleContainer>
            <TupleImage ref={setRef} />
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
