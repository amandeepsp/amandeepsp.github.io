import { useCallback, useRef } from "react"
import p5 from "p5"

export const useP5 = (sketch) => {
    const p5Ref = useRef(null)
    const canvasParentRef = useRef(null)
    const setCanvasParentRef = useCallback(
        (node) => {
            if (node) {
                p5Ref.current = new p5(sketch, node)
            } else {
                if (p5Ref.current) p5Ref.current.remove()
            }
            canvasParentRef.current = node
        },
        [sketch]
    )

    return [setCanvasParentRef, p5Ref.current]
}
