import { pallette } from "../utils/constants"

const minRadius = 2
const maxRadius = 50
const totalCircles = 250
const createCircleAttempts = 500

export const circlePacking = (sketch) => {
    let circles = []

    sketch.setup = function () {
        sketch.createCanvas(200, 200)
        sketch.background(pallette.background)

        for (let i = 0; i < totalCircles; i++) {
            createAndDrawCircle()
        }
    }

    function createAndDrawCircle() {
        let newCircle
        let circleSafeToDraw = false
        for (let tries = 0; tries < createCircleAttempts; tries++) {
            newCircle = {
                x: sketch.random(sketch.width),
                y: sketch.random(sketch.height),
                radius: minRadius,
            }

            if (doesCircleHaveACollision(newCircle)) {
                continue
            } else {
                circleSafeToDraw = true
                break
            }
        }

        if (!circleSafeToDraw) {
            return
        }

        for (let radiusSize = minRadius; radiusSize < maxRadius; radiusSize++) {
            newCircle.radius = radiusSize
            if (doesCircleHaveACollision(newCircle)) {
                newCircle.radius--
                break
            }
        }

        circles.push(newCircle)
        sketch.noStroke()
        sketch.fill(sketch.random(pallette.colors))
        sketch.circle(newCircle.x, newCircle.y, 2 * newCircle.radius)
    }

    function doesCircleHaveACollision(circle) {
        for (let i = 0; i < circles.length; i++) {
            const otherCircle = circles[i]
            const a = circle.radius + otherCircle.radius
            const x = circle.x - otherCircle.x
            const y = circle.y - otherCircle.y

            if (a >= Math.sqrt(x * x + y * y)) {
                return true
            }
        }

        if (circle.x + circle.radius >= sketch.width || circle.x - circle.radius <= 0) {
            return true
        }

        if (circle.y + circle.radius >= sketch.height || circle.y - circle.radius <= 0) {
            return true
        }

        return false
    }
}
