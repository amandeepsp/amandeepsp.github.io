module.exports = {
    content: ["./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}"],
    darkMode: "class",
    plugins: [require("@tailwindcss/typography")],
    theme: {
        extend: {
            fontSize: {
                "4xl": "2.5rem",
                "5xl": "3rem",
                "6xl": "3.5rem",
                "7xl": "4rem"
            }
        }
    }
};
