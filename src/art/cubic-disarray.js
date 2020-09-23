import { format } from "prettier"
import { pallette } from "../utils/constants"

export const cubicDisarray = function (sketch) {
    const squareSize = 20
    const randomDisplacement = 5
    const rotateMultiplier = 5

    sketch.setup = function () {
        sketch.createCanvas(200, 200)
        sketch.background(pallette.background)

        for (let x = squareSize; x < sketch.width - squareSize; x += squareSize) {
            for (let y = squareSize; y < sketch.height - squareSize; y += squareSize) {
                var rotateAmt =
                    (y / sketch.height) * (Math.PI / 180) * sketch.random(-1, 1) * rotateMultiplier

                var translateAmt = (y / sketch.height) * sketch.random(-1, 1) * randomDisplacement

                sketch.push()
                sketch.translate(translateAmt, 0)
                sketch.rotate(rotateAmt)
                sketch.fill(sketch.random(pallette.colors))
                sketch.square(x, y, squareSize)
                sketch.pop()
            }
        }
    }
}
