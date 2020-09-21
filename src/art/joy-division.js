import * as tome from "chromotome"

const pallette = tome.get("cc245")
const step = 10

export const joyDivision = (sketch) => {

    sketch.setup = function(){
        sketch.createCanvas(200, 200)
        sketch.background(pallette.background)
    }
}