import { pallette } from "../utils/constants"

export const joyDivision = (sketch) => {
    const step = 5

    sketch.setup = function () {
        sketch.createCanvas(200, 200)
        sketch.background(pallette.background)
        sketch.fill(pallette.background)
        for (let y = 50; y < sketch.height; y += 2*step) {
            sketch.fill(sketch.random(pallette.colors))
            sketch.beginShape()
            for (let x = 0; x < sketch.width; x += step) {
                sketch.vertex(
                    x,
                    y -
                        (80 / (1 + sketch.pow(x - sketch.width / 2, 4) / 8e6)) *
                            sketch.noise(x / 30 + y)
                )
            }
            sketch.endShape()
        }
    }
}
