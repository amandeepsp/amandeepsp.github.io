import Typography from "typography"
import githubTheme from "typography-theme-github"

const customizations = {
    headerFontFamily: ["noto serif", "serif"],
    bodyFontFamily: ["noto sans", "sans-serif"],
    boldWeight: 700,
    headerWeight: 700,
}

const typography = new Typography({
    ...githubTheme,
    ...customizations,
})

export default typography
