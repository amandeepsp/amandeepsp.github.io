import { useStaticQuery, graphql } from "gatsby"

export const useSiteMetadata = () => {
    const { site } = useStaticQuery(
        graphql`
      query SiteMetaData {
        site {
          siteMetadata {
            title
            description
            author
            social{
                website
                username
            }
            headerNav{
                title
                url
            }
          } 
        }
      }
    `
    )
    return site.siteMetadata
}