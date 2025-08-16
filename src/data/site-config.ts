export type Image = {
    src: string;
    alt?: string;
    caption?: string;
};

export type Link = {
    text: string;
    href: string;
};

export type SiteConfig = {
    title: string;
    subtitle?: string;
    description?: string;
    image?: Image;
    headerNavLinks?: Link[];
    socialLinks?: Link[];
    postsPerPage?: number;
    projectsPerPage?: number;
    author: {
        name: string;
        email: string;
    };
    lang: string;
    locale: string;
};

const siteConfig: SiteConfig = {
    title: "Amandeep Singh",
    author: {
        name: "Amandeep Singh",
        email: "amandeepsp@gmail.com"
    },
    lang: "en",
    locale: "en_US",
    headerNavLinks: [
        {
            text: "Home",
            href: "/"
        },
        {
            text: "Blog",
            href: "/blog"
        },
        {
            text: "Tags",
            href: "/tags"
        },
        {
            text: "Contact Me",
            href: "/contact-me"
        }
    ],
    socialLinks: [
        {
            text: "Github",
            href: "https://github.com/amandeepsp"
        },
        {
            text: "X/Twitter",
            href: "https://x.com/theamndeepsingh"
        },
        {
            text: "LinkedIn",
            href: "https://linkedin.com/in/amandeepsp"
        },
        {
            text: "Email",
            href: "mailto:amandeepsp@gmail.com"
        },
        {
            text: "RSS",
            href: "/rss.xml"
        }
    ],
    postsPerPage: 10,
    projectsPerPage: 10
};

export default siteConfig;
