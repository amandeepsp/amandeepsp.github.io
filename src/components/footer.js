import React from 'react'
import styled from "styled-components"
import { useSiteMetadata } from '../hooks/use-site-metadata'

const Container = styled.div`
    border-top: 1px solid lightblue;
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

export default function Footer() {
    const { social, description, title } = useSiteMetadata()

    const socialLinks = social.map(({ website, username }) => {
        return(
            <SocialItemContainer key={website}>
                <SocialSvg>
                    <use xlinkHref={`social-icon.svg#${website}`}></use>
                </SocialSvg>
                <a href={resolveProfileLink(website, username)} target="_blank" rel="noreferrer">{username}</a>
            </SocialItemContainer>
        )
    })

    return (
        <Container>
            <h4>{title}</h4>
            <SubduedText>{description}</SubduedText>
            <SocialContainer>
                {socialLinks}
            </SocialContainer>
        </Container>
    )
}

const resolveProfileLink = (website, username) => {
    switch(website){
        case 'linkedin':
            return `https://www.linkedin.com/in/${username}`
        case 'stackoverflow':
            return `https://stackoverflow.com/users/${username}`
        default:
            return `https://www.${website}.com/${username}`
    }
}