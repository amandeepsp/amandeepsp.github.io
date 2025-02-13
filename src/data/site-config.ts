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
    logo?: Image;
    title: string;
    subtitle?: string;
    description: string;
    headerNavLinks?: Link[];
    socialLinks?: Link[];
    postsPerPage?: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    title: 'Down the Rabbit Hole',
    subtitle: 'A blog about any and all things engineering',
    description: 'A blog about any and all things engineering',
    headerNavLinks: [
        {
            text: 'Home',
            href: '/'
        },
        {
            text: 'Blog',
            href: '/blog'
        },
        {
            text: 'Tags',
            href: '/tags'
        }
    ],
    socialLinks: [
        {
            text: 'Github',
            href: 'https://github.com/amandeepsp'
        },
        {
            text: 'X/Twitter',
            href: 'https://x.com/theamndeepsingh'
        }
    ],
    postsPerPage: 10,
    projectsPerPage: 10
};

export default siteConfig;
