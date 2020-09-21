import * as tome from "chromotome"

const step = 20
const pallette = tome.get("cc245")

export const tiledLines = (sketch) => {
    sketch.setup = function () {
        sketch.createCanvas(200, 200)
        sketch.background(pallette.background)
        sketch.strokeCap(sketch.ROUND)
        sketch.strokeWeight(5)
        
        for (let x = 0; x < sketch.width; x += step) {
            for (let y = 0; y < sketch.height; y += step) {
                sketch.stroke(sketch.random(pallette.colors))
                lineDraw(x, y, step, step)
            }
        }
    }

    function lineDraw(x, y, width, height) {
        let leftToRight = sketch.random(1) >= 0.5

        if (leftToRight) {
            sketch.line(x, y, x + width, y + height)
        } else {
            sketch.line(x + width, y, x, y + height)
        }
    }
}
