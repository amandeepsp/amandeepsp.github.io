import React from "react"
import styled from "styled-components"
import Header from "./header"
import Footer from "./footer"

const Container = styled.div`
    margin: 2rem auto;
    padding: 0 1rem;
    max-width: 900px;
    display: flex;
    min-height: 95vh;
    flex-direction: column;
`

const ContentContainer = styled.div`
    flex: 1;
`

export default function Layout({ children }) {
    return (
        <Container>
            <Header />
            <ContentContainer>{children}</ContentContainer>
            <Footer />
        </Container>
    )
}
