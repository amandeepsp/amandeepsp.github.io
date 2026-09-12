#import "@preview/cetz:0.5.2": canvas, draw

#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(size: 10pt, fill: rgb("#292524"))

#let ink = rgb("#292524")
#let muted = rgb("#78716c")
#let context-fill = rgb("#f5f5f4")
#let draft-fill = rgb("#dbeafe")
#let draft-stroke = rgb("#3b82f6")
#let accepted-fill = rgb("#dcfce7")
#let accepted-stroke = rgb("#16a34a")
#let rejected-fill = rgb("#fee2e2")
#let rejected-stroke = rgb("#dc2626")
#let pass-fill = rgb("#fafaf9")

#canvas(length: 1cm, {
  import draw: *

  content((0, 4.25), text(fill: muted, weight: "medium")[INPUT], anchor: "west")
  content((0, 3.35), [The cat], name: "context", frame: "rect", padding: (x: 10pt, y: 7pt), fill: context-fill, stroke: ink)

  content((4.05, 4.25), text(fill: draft-stroke, weight: "medium")[DRAFT PROPOSAL], anchor: "west")
  content((4.6, 3.35), [sat], name: "d1", frame: "rect", padding: (x: 10pt, y: 7pt), fill: draft-fill, stroke: draft-stroke)
  content((7, 3.35), [on], name: "d2", frame: "rect", padding: (x: 10pt, y: 7pt), fill: draft-fill, stroke: draft-stroke)
  content((9.4, 3.35), [mat], name: "d3", frame: "rect", padding: (x: 10pt, y: 7pt), fill: draft-fill, stroke: draft-stroke)

  line("context", "d1", stroke: (paint: muted, thickness: .8pt), mark: (end: "stealth"))
  line("d1", "d2", stroke: (paint: draft-stroke, thickness: .8pt))
  line("d2", "d3", stroke: (paint: draft-stroke, thickness: .8pt))

  rect((-0.9, -1.25), (10.75, 2.25), radius: 5pt, fill: pass-fill, stroke: (paint: rgb("#d6d3d1"), thickness: .8pt))

  content((0, 0.72), text(fill: muted)[$p_1(y | "The cat")$], anchor: "west")
  content((4.6, 0.72), [sat #text(fill: accepted-stroke, weight: "bold")[✓]], name: "t1", frame: "rect", padding: (x: 9pt, y: 6pt), fill: accepted-fill, stroke: accepted-stroke)

  content((0, -0.05), text(fill: muted)[$p_2(y | "The cat sat")$], anchor: "west")
  content((7, -0.05), [on #text(fill: accepted-stroke, weight: "bold")[✓]], name: "t2", frame: "rect", padding: (x: 9pt, y: 6pt), fill: accepted-fill, stroke: accepted-stroke)

  content((0, -0.82), text(fill: muted)[$p_3(y | "The cat sat on")$], anchor: "west")
  content((9.4, -0.82), [the #text(fill: rejected-stroke, weight: "bold")[≠ mat]], name: "t3", frame: "rect", padding: (x: 9pt, y: 6pt), fill: rejected-fill, stroke: rejected-stroke)

  line("d1.south", "t1.north", stroke: (paint: accepted-stroke, thickness: 1pt), mark: (end: "stealth"))
  line("d2.south", "t2.north", stroke: (paint: accepted-stroke, thickness: 1pt), mark: (end: "stealth"))
  line("d3.south", "t3.north", stroke: (paint: rejected-stroke, thickness: 1pt), mark: (end: "stealth"))
  content((0, 1.72), text(fill: muted, size: 8pt, weight: "medium")[ONE TARGET-MODEL FORWARD PASS], anchor: "west", fill: pass-fill, padding: 2pt)

  content((0, -2.05), text(fill: muted, weight: "medium")[COMMIT], anchor: "west")
  content((4.6, -2.05), [sat], frame: "rect", padding: (x: 10pt, y: 7pt), fill: accepted-fill, stroke: accepted-stroke)
  content((7, -2.05), [on], frame: "rect", padding: (x: 10pt, y: 7pt), fill: accepted-fill, stroke: accepted-stroke)
  content((9.4, -2.05), [the], frame: "rect", padding: (x: 10pt, y: 7pt), fill: rejected-fill, stroke: rejected-stroke)
})
