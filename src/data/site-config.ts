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
    title: string;
    text: string;
};

export type SiteConfig = {
    logo?: Image;
    title: string;
    subtitle?: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    hero?: Hero;
    socialLinks?: Link[];
    postsPerPage?: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    title: 'Down the Rabbit Hole',
    subtitle: 'A blog about any and all things engineering',
    description: 'A blog about any and all things engineering',
    logo: {
        src: '/alembic_2697-fe0f.png'
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
    hero: {
        title: 'Hi! I am Amandeep',
        text: `I have been a software developer for almost 8 years,
        starting out as an android app developer, then branching out to both frontend and backend.
        This blog is where I explore the inner workings of software, hardware, and whatever else catches my curiosity.
        If you like peeling back the layers of how things work, stick around — there's plenty to dig into.
        \nTo get in touch, my social media handles are at the bottom of the page.`
    },
    socialLinks: [
        {
            text: 'Github',
            href: 'https://github.com/amandeepsp'
        },
        {
            text: 'X/Twitter',
            href: 'https://x.com/theamndeepsingh'
        },
        {
            text: 'LinkedIn',
            href: 'https://linkedin.com/in/amandeepsp'
        },
        {
            text: 'Email',
            href: 'mailto:amandeepsp@gmail.com'
        },
        {
            text: 'RSS',
            href: '/rss.xml'
        }
    ],
    postsPerPage: 10,
    projectsPerPage: 10
};

export default siteConfig;
