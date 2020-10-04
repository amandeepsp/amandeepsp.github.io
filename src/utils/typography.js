import Typography from "typography"
import githubTheme from "typography-theme-github"

const customizations = {
    headerFontFamily: ["Rubik", "sans-serif"],
    bodyFontFamily: ["Karla", "sans-serif"],
}

const typography = new Typography({
    ...githubTheme,
    ...customizations,
})

export default typography
