import React from "react"
import kebabCase from "lodash/kebabCase"
import sample from "lodash/sample"
import { circlePacking, tiledLines, cubicDisarray } from "../art"
import { useP5 } from "../hooks/use-p5"
import { Link } from "gatsby"
import styled from "styled-components"

const TupleContainer = styled.div`
    margin: 1rem 1rem 1rem 0;
    width: 200px;
`


export default function TagTuple({ tag }) {
    const sketch = sample([tiledLines, circlePacking, cubicDisarray])
    const [setRef] = useP5(sketch)
    return (
        <TupleContainer>
            <div ref={setRef}/>
            <Link key={tag.fieldValue} to={`/tags/${kebabCase(tag.fieldValue)}/`}>
                {tag.fieldValue} ({tag.totalCount})
            </Link>
        </TupleContainer>
    )
}
