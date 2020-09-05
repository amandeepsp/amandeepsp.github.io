import React from "react"
import styled from "styled-components"
import { Link } from "gatsby"
import { useSiteMetadata } from "../hooks/use-site-metadata"
import { MOBILE_QUERY_SIZE } from "../utils/constants"

const Container = styled.div`
    display: flex;
    flex-direction: row;
    @media (max-width: ${MOBILE_QUERY_SIZE}px) {
        flex-direction: column;
    }
    align-items: baseline;
    border-bottom: 1px solid gray;
`

const NavContainer = styled.div`
    display: flex;
    flex-direction: row;
`

const SiteTitle = styled.h1`
    flex-grow: 1;
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
`
const NavLink = styled.h3`
    margin: 0 0.5rem;
    @media (max-width: ${MOBILE_QUERY_SIZE}px) {
        margin: 0 0.75rem 0.5rem 0;
    }
`

export default function Header() {
    const { title, headerNav } = useSiteMetadata()

    const navItems = headerNav.map(({ title, url }) => {
        return (
            <NavLink key={url}>
                <Link to={url}>{title}</Link>
            </NavLink>
        )
    })

    return (
        <Container>
            <SiteTitle>
                <Link to={"/"}>{title}</Link>
            </SiteTitle>
            <NavContainer>{navItems}</NavContainer>
        </Container>
    )
}
