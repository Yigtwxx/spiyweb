import { useEffect, useMemo, useRef, useState } from "react";
import type { SceneDto, SceneEdgeDto, SceneNodeDto } from "../lib/types";

/**
 * The activated web.
 *
 * SVG, not canvas. At the default cap (300 atoms, contributor edges) the DOM
 * cost is small, and what SVG buys is not cosmetic: every atom is a focusable
 * element with a label, so the web can be read by keyboard and by a screen
 * reader. Edges are batched into one path per layer, so the induced mode
 * cannot explode the node count.
 *
 * Two layouts share one determinism contract, and both come from the Python
 * side so the Streamlit tool and this page draw the same picture:
 *   force — what is connected to what
 *   hops  — how far the energy actually got
 */

const KIND_FILL: Record<string, string> = {
  seed: "var(--color-energy)",
  bridge: "var(--color-destroyed)",
  activated: "var(--color-structure)",
  suppressed: "var(--color-ink-quiet)",
};

type Layout = "force" | "hops";

// The drawing box is 16:10, so a square viewBox would be letterboxed into the
// middle 62.5% of the width - a third of the canvas lost before a single atom
// is placed. The force layout is therefore mapped onto the full box, while the
// hop layout keeps a centred SQUARE: its rings are circles, and stretching the
// positions without stretching the rings would put atoms off their own band.
const W = 1600;
const H = 1000;
const RING_INSET = (W - H) / 2;

// Fit constants, in the same units as W and H.
/** Breathing room around the drawing, so atoms never touch the plate edge. */
const FIT_PADDING = 70;
/** Room above an atom for its label, which the circle's radius does not cover. */
const LABEL_HEADROOM = 26;
/** Smallest span the fit will zoom to. A single activated atom must not fill
 *  the plate: a circle's size is its energy, and one alone has nothing to be
 *  compared against, so blowing it up would state a magnitude that is not there. */
const FIT_MIN_SPAN = W * 0.42;

export function WebCanvas({
  scene,
  layout,
  animate,
  selected,
  onSelect,
}: {
  scene: SceneDto;
  layout: Layout;
  animate: boolean;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const [focusIndex, setFocusIndex] = useState(0);
  const [replay, setReplay] = useState(0);
  const groupRef = useRef<SVGGElement>(null);

  useEffect(() => setReplay((value) => value + 1), [scene, layout]);

  const nodes = scene.nodes;
  const maxEnergy = useMemo(
    () => Math.max(1e-9, ...nodes.map((node) => node.energy)),
    [nodes],
  );

  const project = (x: number, y: number): [number, number] =>
    layout === "force" ? [x * W, y * H] : [RING_INSET + x * H, y * H];
  const at = (node: SceneNodeDto): [number, number] =>
    layout === "force" ? project(node.x, node.y) : project(node.rx, node.ry);
  const edgeAt = (edge: SceneEdgeDto): [number, number, number, number] => {
    const [x1, y1] =
      layout === "force"
        ? project(edge.x1, edge.y1)
        : project(edge.rx1, edge.ry1);
    const [x2, y2] =
      layout === "force"
        ? project(edge.x2, edge.y2)
        : project(edge.rx2, edge.ry2);
    return [x1, y1, x2, y2];
  };

  const radius = (node: SceneNodeDto) =>
    node.kind === "suppressed"
      ? 4
      : 5 + 13 * Math.sqrt(node.energy / maxEnergy);

  const radiusOf = useMemo(() => {
    const table = new Map<string, number>();
    for (const node of nodes) table.set(node.id, radius(node));
    return table;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, maxEnergy]);

  // Painter's order, not array order. The scene lists atoms strongest first,
  // so the largest circles were drawn first and every faint neighbour landed
  // on top of them. Ghosts go down first, then by energy, so the atoms that
  // hold the energy end up in front of the ones that barely lit.
  const drawOrder = useMemo(
    () =>
      nodes
        .map((_, index) => index)
        .sort((a, b) => {
          const ghost = (i: number) => (nodes[i].kind === "suppressed" ? 0 : 1);
          return (
            ghost(a) - ghost(b) || nodes[a].energy - nodes[b].energy || a - b
          );
        }),
    [nodes],
  );

  // Labels are placed by energy, and one that would land on an already
  // placed one is dropped rather than stacked. In the dense middle of a web
  // three ids used to overlap into an unreadable smudge, which is worse than
  // showing two of them: the reader could not recover any of the three.
  const visibleLabels = useMemo(() => {
    const boxes: [number, number, number, number][] = [];
    const kept = new Set<string>();
    const byEnergy = [...nodes]
      .filter((node) => node.label)
      .sort((a, b) => b.energy - a.energy || a.id.localeCompare(b.id));
    for (const node of byEnergy) {
      const [x, y] = at(node);
      const halfWidth = (node.label.length * 6.6) / 2;
      const top = y - radius(node) - 18;
      const box: [number, number, number, number] = [
        x - halfWidth,
        top,
        x + halfWidth,
        top + 14,
      ];
      const clash = boxes.some(
        (other) =>
          box[0] < other[2] &&
          box[2] > other[0] &&
          box[1] < other[3] &&
          box[3] > other[1],
      );
      if (clash) continue;
      boxes.push(box);
      kept.add(node.id);
    }
    return kept;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, layout, maxEnergy]);

  // Everything the links needed and did not have:
  //  - they were drawn centre to centre, so each one disappeared under the
  //    two atoms it connected and a 20px seed swallowed a dozen of them;
  //  - they were dead straight, so parallel and crossing links piled into one
  //    unreadable smear;
  //  - they all had the same weight on screen, so a link carrying most of the
  //    energy looked exactly like one carrying a trace.
  const byLayer = useMemo(() => {
    const drawn = scene.edges.filter((edge) => edge.kind !== "suppressed");
    const sorted = [...drawn].map((edge) => edge.weight).sort((a, b) => a - b);
    // Top quartile is drawn heavier. A quantile, not a fixed number: edge
    // weights are layer-dependent and an absolute cut would mark every entity
    // edge and no semantic one.
    const strongAt = sorted.length
      ? sorted[Math.floor(sorted.length * 0.75)]
      : Infinity;

    const grouped: Record<
      string,
      { active: string[]; strong: string[]; cut: string[] }
    > = {};
    for (const edge of scene.edges) {
      const [x1, y1, x2, y2] = edgeAt(edge);
      const dx = x2 - x1;
      const dy = y2 - y1;
      const length = Math.hypot(dx, dy);
      grouped[edge.layer] ??= { active: [], strong: [], cut: [] };
      if (length < 1) continue;
      const ux = dx / length;
      const uy = dy / length;
      const start = (radiusOf.get(edge.source) ?? 0) + 2;
      const end = (radiusOf.get(edge.target) ?? 0) + 2;
      // Two atoms almost touching leave no room for a link between them;
      // drawing one anyway just thickens their outlines.
      if (length <= start + end + 3) continue;
      const ax = x1 + ux * start;
      const ay = y1 + uy * start;
      const bx = x2 - ux * end;
      const by = y2 - uy * end;
      // One consistent bow direction: arcs that all lean the same way read as
      // a drawing convention, while alternating signs read as noise.
      const bow = Math.min(0.09 * length, 55);
      const cx = (ax + bx) / 2 - uy * bow;
      const cy = (ay + by) / 2 + ux * bow;
      const command = `M${ax.toFixed(1)} ${ay.toFixed(1)}Q${cx.toFixed(
        1,
      )} ${cy.toFixed(1)} ${bx.toFixed(1)} ${by.toFixed(1)}`;
      if (edge.kind === "suppressed") grouped[edge.layer].cut.push(command);
      else if (edge.weight >= strongAt)
        grouped[edge.layer].strong.push(command);
      else grouped[edge.layer].active.push(command);
    }
    return grouped;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, layout, radiusOf]);

  // One tab stop for the whole canvas, arrow keys inside: tabbing through
  // three hundred atoms would be punishment, not access.
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (nodes.length === 0) return;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      setFocusIndex((index) => (index + 1) % nodes.length);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      setFocusIndex((index) => (index - 1 + nodes.length) % nodes.length);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(nodes[focusIndex]?.id ?? null);
    } else if (event.key === "Escape") {
      onSelect(null);
    }
  };

  const focused = nodes[focusIndex];

  // The viewBox is fitted to the drawing, not fixed to the sheet.
  //
  // Both layouts normalise into [0,1] and then place the atoms wherever the
  // web's shape puts them — which, for a query that activated eight atoms
  // across two hops, is a sixty-pixel knot in the middle of a nine-hundred
  // pixel plate. The picture was three percent of the space it was given and
  // the labels collapsed into each other, so the page's own subject read as
  // an empty sheet.
  //
  // Fitting means measuring what was actually drawn — atoms expanded by their
  // radius, plus room above for a label — and pointing the viewBox at that.
  // `FIT_MIN_SPAN` stops the other extreme: a single activated atom must not
  // be blown up to fill the plate, because the size of a circle is the size
  // of its energy and a lone one carries no comparison to be scaled against.
  const fit = useMemo(() => {
    if (!nodes.length) return { x: 0, y: 0, w: W, h: H };
    let left = Infinity;
    let right = -Infinity;
    let top = Infinity;
    let bottom = -Infinity;
    for (const node of nodes) {
      const [x, y] = at(node);
      const r = radius(node);
      left = Math.min(left, x - r);
      right = Math.max(right, x + r);
      // Labels sit above their atom; the drawing is taller than its circles.
      top = Math.min(top, y - r - LABEL_HEADROOM);
      bottom = Math.max(bottom, y + r);
    }
    const pad = FIT_PADDING;
    left -= pad;
    right += pad;
    top -= pad;
    bottom += pad;

    // Grow the tighter axis so the sheet's proportion is preserved and the
    // drawing lands centred rather than stretched.
    let width = Math.max(right - left, FIT_MIN_SPAN);
    let height = Math.max(bottom - top, (FIT_MIN_SPAN * H) / W);
    const ratio = W / H;
    if (width / height < ratio) width = height * ratio;
    else height = width / ratio;

    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    return { x: cx - width / 2, y: cy - height / 2, w: width, h: height };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, layout, maxEnergy]);

  return (
    <div className="relative">
      <svg
        viewBox={`${fit.x} ${fit.y} ${fit.w} ${fit.h}`}
        className="w-full"
        style={{ aspectRatio: "16 / 10", touchAction: "none" }}
        role="application"
        tabIndex={0}
        aria-label={`Activated web: ${nodes.length} atoms, ${scene.edges.length} links. Arrow keys move between atoms, Enter opens details.`}
        onKeyDown={onKeyDown}
      >
        <defs>
          {/* The grid is the scale of the drawing, so it belongs inside the
              coordinate system and moves with it — a CSS background would
              stay put while the content moved. */}
          <pattern
            id="grid-minor"
            width="20"
            height="20"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M20 0H0V20"
              fill="none"
              stroke="var(--color-navy-600)"
              strokeWidth="1"
            />
          </pattern>
          <pattern
            id="grid-major"
            width="120"
            height="120"
            patternUnits="userSpaceOnUse"
          >
            <rect width="120" height="120" fill="url(#grid-minor)" />
            <path
              d="M120 0H0V120"
              fill="none"
              stroke="var(--color-navy-500)"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#grid-major)" />

        {/* Rings belong to the hop layout only. They were drawn in both so
            that hop depth would not live solely in an animation, but in the
            force layout a node's distance from the centre is set by the
            spring solver, not by its hop - so concentric gold bands promised
            a reading the picture could not support. Hop depth survives the
            change: every atom's tooltip states it, the spread animation
            paces on it, and the hop layout - where the rings do mean what
            they look like - is one click away. */}
        <g aria-hidden="true">
          {(layout === "hops"
            ? (scene.rings ?? []).filter((ring) => ring.hop >= 0)
            : []
          ).map(({ hop, radius }) => {
            // The radius arrives with the scene. This used to be a second
            // copy of the server's spacing formula, and when that rule
            // changed the guides drew a layout that no longer existed —
            // circles in empty space with the atoms somewhere else.
            const r = radius * H;
            return (
              <g key={hop}>
                <circle
                  cx={W / 2}
                  cy={H / 2}
                  r={r}
                  fill="none"
                  stroke="var(--color-rule-faint)"
                  strokeWidth="1"
                />
                <text
                  x={W / 2 + r + 6}
                  y={H / 2 + 2}
                  fontSize="13"
                  fill="var(--color-rule)"
                  fontFamily="var(--font-mono)"
                >
                  hop {hop}
                </text>
              </g>
            );
          })}
        </g>

        <g ref={groupRef} key={replay}>
          {Object.entries(byLayer).map(([layer, paths]) => {
            const colour = scene.legend[layer] ?? "var(--color-structure)";
            return (
              <g key={layer} strokeLinecap="round">
                {paths.active.length ? (
                  <path
                    d={paths.active.join("")}
                    stroke={colour}
                    strokeWidth="1.1"
                    strokeOpacity="0.34"
                    fill="none"
                  />
                ) : null}
                {paths.strong.length ? (
                  <path
                    d={paths.strong.join("")}
                    stroke={colour}
                    strokeWidth="2.2"
                    strokeOpacity="0.7"
                    fill="none"
                  />
                ) : null}
                {paths.cut.length ? (
                  <path
                    d={paths.cut.join("")}
                    stroke={colour}
                    strokeWidth="2"
                    strokeDasharray="7 5"
                    strokeOpacity="0.95"
                    fill="none"
                  />
                ) : null}
              </g>
            );
          })}

          {drawOrder.map((index) => {
            const node = nodes[index];
            const [x, y] = at(node);
            const isFocused = index === focusIndex;
            const isSelected = node.id === selected;
            return (
              <g
                key={node.id}
                className={animate ? "atom" : undefined}
                style={
                  { "--hop": Math.max(0, node.hop) } as React.CSSProperties
                }
                onClick={() => {
                  setFocusIndex(index);
                  onSelect(node.id);
                }}
              >
                <title>{node.tooltip}</title>
                {node.kind === "bridge" ? (
                  <rect
                    x={x - radius(node)}
                    y={y - radius(node)}
                    width={radius(node) * 2}
                    height={radius(node) * 2}
                    transform={`rotate(45 ${x} ${y})`}
                    fill={KIND_FILL[node.kind]}
                    stroke="var(--color-navy-900)"
                    strokeWidth="1.6"
                  />
                ) : (
                  <circle
                    cx={x}
                    cy={y}
                    r={radius(node)}
                    fill={KIND_FILL[node.kind] ?? KIND_FILL.activated}
                    fillOpacity={node.kind === "suppressed" ? 0.4 : 1}
                    stroke={
                      node.kind === "seed"
                        ? "var(--color-rule)"
                        : "var(--color-navy-900)"
                    }
                    strokeWidth={node.kind === "seed" ? 2.5 : 1.6}
                    strokeDasharray={
                      node.kind === "suppressed" ? "3 3" : undefined
                    }
                  />
                )}
                {isSelected || isFocused ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={radius(node) + 6}
                    fill="none"
                    stroke="var(--color-rule)"
                    strokeWidth={isSelected ? 2 : 1}
                    strokeDasharray={isSelected ? undefined : "3 3"}
                  />
                ) : null}
                {node.label && visibleLabels.has(node.id) ? (
                  <text
                    x={x}
                    y={y - radius(node) - 7}
                    textAnchor="middle"
                    fontSize="12"
                    fill="var(--color-ink-quiet)"
                    fontFamily="var(--font-mono)"
                    stroke="var(--color-navy-900)"
                    strokeWidth="3.5"
                    strokeLinejoin="round"
                    // The id has to survive being read on top of a mesh of
                    // links. `paint-order` puts the knockout stroke behind
                    // the glyphs, which costs one attribute and no DOM.
                    style={{ paintOrder: "stroke" }}
                  >
                    {node.label}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>

      {focused ? (
        <p className="sr-only" aria-live="polite">
          {focused.id}, energy {focused.energy.toFixed(3)}, hop {focused.hop},{" "}
          {focused.votes} votes
        </p>
      ) : null}
    </div>
  );
}
