import React from "react"
import styled from "styled-components"
import Header from "./header"
import Footer from "./footer"

const Container = styled.div`
    margin: 3rem auto;
    padding: 0 1rem;
    max-width: 900px;
	min-height: 100%;

`

export default function Layout({ children }) {
	return (
		<Container>
			<Header />
			{children}
			<Footer />
		</Container>
	)
}