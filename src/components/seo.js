import React from "react"
import { Helmet } from "react-helmet"
import { useSiteMetadata } from "../hooks/use-site-metadata"
import PropTypes from "prop-types"

export default function SEO({ lang, description, title, meta, image: metaImage }) {
    const siteMetadata = useSiteMetadata()

    const metaDescription = description || siteMetadata.description

    const image = metaImage && metaImage.src ? `${siteMetadata.siteUrl}${metaImage.src}` : null

    const customMeta = [
        {
            name: `description`,
            content: metaDescription,
        },
        {
            property: `og:title`,
            content: title,
        },
        {
            property: `og:description`,
            content: metaDescription,
        },
        {
            property: `og:type`,
            content: `website`,
        },
        {
            name: `twitter:card`,
            content: `summary`,
        },
        {
            name: `twitter:creator`,
            content: siteMetadata.author,
        },
        {
            name: `twitter:title`,
            content: title,
        },
        {
            name: `twitter:description`,
            content: metaDescription,
        },
    ].concat(
        metaImage
            ? [
                  {
                      property: "og:image",
                      content: image,
                  },
                  {
                      property: "og:image:width",
                      content: metaImage.width,
                  },
                  {
                      property: "og:image:height",
                      content: metaImage.height,
                  },
                  {
                      name: "twitter:card",
                      content: "summary_large_image",
                  },
              ]
            : [
                  {
                      name: "twitter:card",
                      content: "summary",
                  },
              ]
    )
    return (
        <Helmet
            htmlAttributes={{
                lang,
            }}
            title={title}
            titleTemplate={`%s | ${siteMetadata.title}`}
            meta={customMeta.concat(meta)}
        />
    )
}

SEO.defaultProps = {
    lang: `en`,
    meta: [],
}

SEO.propTypes = {
    description: PropTypes.string,
    lang: PropTypes.string,
    meta: PropTypes.arrayOf(PropTypes.object),
    title: PropTypes.string.isRequired,
    image: PropTypes.shape({
        src: PropTypes.string.isRequired,
        height: PropTypes.number.isRequired,
        width: PropTypes.number.isRequired,
    }),
}
