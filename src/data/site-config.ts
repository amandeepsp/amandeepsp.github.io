export type Image = {
    src: string;
    alt?: string;
    caption?: string;
};

export type Link = {
    text: string;
    href: string;
};

export type Hero = {
    title?: string;
    text?: string;
};

export type SiteConfig = {
    logo?: Image;
    title: string;
    subtitle?: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    socialLinks?: Link[];
    hero?: Hero;
    postsPerPage?: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    title: 'Down the Rabbit Hole',
    subtitle: 'A Blog about software engineering',
    description: 'A blog about software engineering, AI, and the future',
    image: {
        src: '',
        alt: 'Down the Rabbit Hole - A blog about software engineering, AI, and the future'
    },
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
    hero: {
        title: 'Hi There & Welcome to My Corner of the Web!',
        text: "I'm **Amandeep**, a software engineer."
    },
    postsPerPage: 8,
    projectsPerPage: 8
};

export default siteConfig;
