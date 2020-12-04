import React from "react"
import styled from "styled-components"
import { useSiteMetadata } from "../hooks/use-site-metadata"

const Container = styled.div`
    border-top: 1px solid gray;
`
const SubduedText = styled.p`
    color: gray;
    max-width: 500px;
`
const SocialContainer = styled.div`
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
`

const SocialItemContainer = styled.div`
    display: flex;
    flex-direction: row;
    align-items: center;
    margin: 0 1rem 0 0;
`

const SocialSvg = styled.svg`
    width: 24px;
    height: 18px;
    display: inline-block;
    fill: gray;
    padding: 0 4px 0 0;
    vertical-align: text-bottom;
`

const SmallSiteTitle = styled.h4`
    font-family: "noto serif", sans-serif;
    font-weight: 900;
    color: black;
`

export default function Footer() {
    const { social, description, title } = useSiteMetadata()

    const socialLinks = social.map(({ website, username }) => {
        return (
            <SocialItemContainer key={website}>
                <SocialSvg>
                    <use xlinkHref={`/social-icon.svg#${website}`}></use>
                </SocialSvg>
                <a
                    href={resolveProfileLink(website, username)}
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    {username}
                </a>
            </SocialItemContainer>
        )
    })

    return (
        <Container>
            <SmallSiteTitle>{title}</SmallSiteTitle>
            <SubduedText>{description}</SubduedText>
            <SocialContainer>{socialLinks}</SocialContainer>
            <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
                <img
                    alt="Creative Commons Licence"
                    style={{ borderWidth: 0, marginTop: "1rem" }}
                    src="https://i.creativecommons.org/l/by/4.0/88x31.png"
                />
            </a>
            <p style={{ fontSize: "0.75rem" }}>
                This work is licensed under a{" "}
                <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
                    Creative Commons Attribution 4.0 International License
                </a>
                .
            </p>
        </Container>
    )
}

const resolveProfileLink = (website, username) => {
    switch (website) {
        case "linkedin":
            return `https://www.linkedin.com/in/${username}`
        case "stackoverflow":
            return `https://stackoverflow.com/users/${username}`
        default:
            return `https://www.${website}.com/${username}`
    }
}
