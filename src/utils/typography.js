import Typography from "typography"
import githubTheme from 'typography-theme-github'
import CodePlugin from 'typography-plugin-code'

githubTheme.baseFontSize = '16px'
githubTheme.overrideThemeStyles = () => ({
    'h1,h2,h3,h4': {
        borderBottom: '0px',
    },
})

githubTheme.plugins = [
    new CodePlugin(),
]

const typography = new Typography(githubTheme)

export default typography