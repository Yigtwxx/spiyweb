/* The hero's live web: an orb web spun across the navy field, continuing the
   engraved one in the plate, with a query's energy running through it.

   The pulse follows the library's own rules (CLAUDE.md §2.1): a seed of 10.0
   lands on one node; every lit node forwards `damping` of what it holds to the
   neighbours it has not lit yet, split in proportion to thread weight; energy
   arriving by two paths adds up; a node that would receive less than the floor
   stays dark. Only the floor differs from the library's 15%: on a web this dense
   the true floor stops the pulse two threads out, which is the point of the
   method and too small to see, so the picture uses `FLOOR` and the trace further
   down the page carries the real numbers.

   On load the web is spun from the hub outward. Between queries, points of light
   drift along the silk and the web sways on springs in a slow wind. A seed lands
   every few seconds somewhere on the web, and wherever the reader clicks
   the field, throwing a ring where it lands. The loop sleeps off-screen and in a hidden tab, and the pause button
   stops it (WCAG 2.2.2). Under reduced motion one pulse is drawn, finished and
   still. */

const SEED = 10;
const DAMPING = 0.6;
const FLOOR = 0.12; // see above: the picture's floor, not the library's
const HOP_MS = 420; // one hop's travel along its threads
const FADE_MS = 3200; // how long a lit node takes to go dark again
const AUTO_MS = 3200; // a seed lands this often when nobody clicks
const RADIALS = 18;
const RINGS = 13;
const SPIN_MS = 2600; // the web is spun from the hub outward when the page opens
const RIPPLE_MS = 1100; // the ring a seed throws when it lands
const MOTES = 16; // points of light drifting along the silk between queries
const MOTE_SPEED = 0.05; // pixels per millisecond
const WIND = 1.4; // the web's idle sway, in pixels

interface WebNode {
    x: number;
    y: number;
    edges: number[]; // indices into `threads`
}
interface Thread {
    a: number;
    b: number;
    w: number; // weight: how much of a split it takes
    radial: boolean;
}
interface Mote {
    thread: number;
    forward: boolean;
    t: number; // 0..1 along the thread
    born: number;
    life: number;
}
interface Lit {
    energy: number;
    hop: number;
    at: number; // when the energy arrived (ms, performance.now)
    from: number; // node it came from, -1 for the seed
}
interface Pulse {
    started: number;
    lit: Map<number, Lit>;
    hops: number;
}

/** A seeded LCG, so the web has the same shape on every load. */
function random(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 2 ** 32;
    };
}

function spin(
    width: number,
    height: number,
    hubX: number,
    hubY: number,
): { nodes: WebNode[]; threads: Thread[] } {
    const rand = random(7);
    const nodes: WebNode[] = [];
    const threads: Thread[] = [];
    const link = (a: number, b: number, w: number): void => {
        const i = threads.push({ a, b, w, radial: w >= 0.7 }) - 1;
        nodes[a]?.edges.push(i);
        nodes[b]?.edges.push(i);
    };
    // Far enough to reach every corner of the field from the hub.
    const reach = Math.hypot(Math.max(hubX, width - hubX), Math.max(hubY, height - hubY)) * 1.05;
    nodes.push({ x: hubX, y: hubY, edges: [] });
    const angles = Array.from(
        { length: RADIALS },
        (_, i) => ((i + rand() * 0.45) / RADIALS) * Math.PI * 2,
    );
    const grid: number[][] = [];
    for (let r = 0; r < RINGS; r++) {
        const ring: number[] = [];
        // Rings open out like a real orb web: tight at the hub, wide at the edge.
        const base = reach * Math.pow((r + 1) / RINGS, 1.35);
        for (let k = 0; k < RADIALS; k++) {
            const angle = angles[k] ?? 0;
            const sag = 1 + (rand() - 0.5) * 0.12;
            ring.push(
                nodes.push({
                    x: hubX + Math.cos(angle) * base * sag,
                    y: hubY + Math.sin(angle) * base * sag,
                    edges: [],
                }) - 1,
            );
        }
        grid.push(ring);
    }
    for (let k = 0; k < RADIALS; k++) {
        // Radials are the strong threads: they carry the energy outward.
        let prev = 0;
        for (let r = 0; r < RINGS; r++) {
            const id = grid[r]?.[k];
            if (id === undefined) continue;
            link(prev, id, 0.7 + rand() * 0.3);
            prev = id;
        }
    }
    for (let r = 0; r < RINGS; r++) {
        // The spiral: weaker, and a few segments missing, as on any real web.
        for (let k = 0; k < RADIALS; k++) {
            if (r > 2 && rand() < 0.12) continue;
            const a = grid[r]?.[k];
            const b = grid[r]?.[(k + 1) % RADIALS];
            if (a !== undefined && b !== undefined) link(a, b, 0.25 + rand() * 0.35);
        }
    }
    return { nodes, threads };
}

/** One query through the web, hop by hop: the library's propagation, by hand. */
function propagate(nodes: WebNode[], threads: Thread[], seed: number, now: number): Pulse {
    const lit = new Map<number, Lit>();
    lit.set(seed, { energy: SEED, hop: 0, at: now, from: -1 });
    let frontier = [seed];
    let hop = 0;
    while (frontier.length > 0 && hop < 40) {
        hop += 1;
        const arriving = new Map<number, { energy: number; from: number }>();
        for (const id of frontier) {
            const node = nodes[id];
            const held = lit.get(id)?.energy ?? 0;
            if (!node) continue;
            const out = node.edges
                .map((e) => threads[e])
                .filter((t): t is Thread => t !== undefined)
                .map((t) => ({ to: t.a === id ? t.b : t.a, w: t.w }))
                .filter(({ to }) => !lit.has(to));
            const total = out.reduce((sum, o) => sum + o.w, 0);
            if (total === 0) continue;
            const forward = held * DAMPING;
            for (const { to, w } of out) {
                const prev = arriving.get(to);
                const share = (forward * w) / total;
                // Converging evidence: two paths into one node add up.
                arriving.set(to, {
                    energy: (prev?.energy ?? 0) + share,
                    from: prev && prev.energy > share ? prev.from : id,
                });
            }
        }
        frontier = [];
        for (const [id, { energy, from }] of arriving) {
            if (energy < FLOOR) continue; // below the floor: it never lights
            lit.set(id, { energy, hop, at: now + hop * HOP_MS, from });
            frontier.push(id);
        }
    }
    return { started: now, lit, hops: hop };
}

export function initWebCanvas(): void {
    const field = document.querySelector<HTMLElement>('[data-web-field]');
    const canvas = field?.querySelector<HTMLCanvasElement>('[data-web-canvas]');
    const ctx = canvas?.getContext('2d');
    if (!field || !canvas || !ctx) return;
    const hub = field.querySelector<HTMLElement>('[data-web-hub]');
    const toggle = field.querySelector<HTMLButtonElement>('[data-web-toggle]');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let nodes: WebNode[] = [];
    let threads: Thread[] = [];
    let pulses: Pulse[] = [];
    let width = 0;
    let height = 0;
    let frame = 0;
    let visible = true;
    let paused = false;
    let lastSeed = 0;
    let spunAt = performance.now();
    let firstSeed = false;
    // Spring state per node: offset from rest and velocity, moved by the wind.
    let ox = new Float32Array(0);
    let oy = new Float32Array(0);
    let vx = new Float32Array(0);
    let vy = new Float32Array(0);
    let reach = 1;
    let motes: Mote[] = [];
    let lastFrame = performance.now();

    const nearest = (x: number, y: number): number => {
        let best = 0;
        let bestD = Infinity;
        nodes.forEach((n, i) => {
            const d = (n.x - x) ** 2 + (n.y - y) ** 2;
            if (d < bestD) {
                bestD = d;
                best = i;
            }
        });
        return best;
    };

    const build = (): void => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const box = field.getBoundingClientRect();
        width = box.width;
        height = box.height;
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // The hub sits on the engraved spider, so the drawn web continues the cut one.
        let hx = width * 0.72;
        let hy = height * 0.45;
        if (hub) {
            const h = hub.getBoundingClientRect();
            hx = h.left - box.left + h.width * 0.5;
            hy = h.top - box.top + h.height * 0.36;
        }
        ({ nodes, threads } = spin(width, height, hx, hy));
        pulses = [];
        ox = new Float32Array(nodes.length);
        oy = new Float32Array(nodes.length);
        vx = new Float32Array(nodes.length);
        vy = new Float32Array(nodes.length);
        reach = Math.max(1, ...nodes.map((n) => Math.hypot(n.x - hx, n.y - hy)));
    };

    /** One spring step: the wind sets a target offset, springs follow it. */
    const settle = (now: number): void => {
        nodes.forEach((n, i) => {
            // Idle sway: a slow breath of wind, stronger towards the rim.
            const rim = Math.min(
                1,
                Math.hypot(n.x - (nodes[0]?.x ?? 0), n.y - (nodes[0]?.y ?? 0)) / reach,
            );
            const tx = Math.sin(now / 1900 + n.y * 0.012) * WIND * rim;
            const ty = Math.cos(now / 2300 + n.x * 0.01) * WIND * 0.6 * rim;
            vx[i] = ((vx[i] ?? 0) + (tx - (ox[i] ?? 0)) * 0.09) * 0.82;
            vy[i] = ((vy[i] ?? 0) + (ty - (oy[i] ?? 0)) * 0.09) * 0.82;
            ox[i] = (ox[i] ?? 0) + (vx[i] ?? 0);
            oy[i] = (oy[i] ?? 0) + (vy[i] ?? 0);
        });
    };
    const px = (i: number): number => (nodes[i]?.x ?? 0) + (ox[i] ?? 0);
    const py = (i: number): number => (nodes[i]?.y ?? 0) + (oy[i] ?? 0);

    const newMote = (now: number): Mote => ({
        thread: Math.floor(Math.random() * threads.length),
        forward: Math.random() < 0.5,
        t: Math.random(),
        born: now,
        life: 5000 + Math.random() * 6000,
    });
    /** Motes run along a thread and, at its end, turn onto a neighbouring one. */
    const drift = (now: number, dt: number): void => {
        while (motes.length < MOTES && threads.length)
            motes.push(newMote(now - Math.random() * 4000));
        motes = motes.map((m) => {
            if (now - m.born > m.life) return newMote(now);
            const t = threads[m.thread];
            if (!t) return newMote(now);
            const len = Math.max(1, Math.hypot(px(t.b) - px(t.a), py(t.b) - py(t.a)));
            let next = m.t + ((m.forward ? 1 : -1) * MOTE_SPEED * dt) / len;
            if (next >= 0 && next <= 1) return { ...m, t: next };
            const at = next > 1 ? t.b : t.a;
            const choices = (nodes[at]?.edges ?? []).filter((e) => e !== m.thread);
            const pick = choices[Math.floor(Math.random() * choices.length)];
            const nt = pick !== undefined ? threads[pick] : undefined;
            if (pick === undefined || !nt) return newMote(now);
            next = 0;
            return { ...m, thread: pick, forward: nt.a === at, t: nt.a === at ? 0 : 1 };
        });
    };

    const seed = (id: number, now: number): void => {
        pulses.push(propagate(nodes, threads, id, now));
        if (pulses.length > 4) pulses.shift();
        lastSeed = now;
    };

    const glow = (pulse: Pulse, id: number, now: number): number => {
        const l = pulse.lit.get(id);
        if (!l || now < l.at) return 0;
        const age = now - l.at;
        const fade = Math.max(0, 1 - age / FADE_MS);
        return Math.min(1, Math.sqrt(l.energy / SEED)) * fade;
    };

    const draw = (now: number): void => {
        ctx.clearRect(0, 0, width, height);
        ctx.lineCap = 'round';
        const hub = nodes[0];
        const spun = Math.min(1, (now - spunAt) / SPIN_MS);
        const radius = reach * 1.05 * (1 - Math.pow(1 - spun, 3)); // ease-out-cubic
        const reachOf = (a: number, b: number): number =>
            Math.max(
                Math.hypot((nodes[a]?.x ?? 0) - (hub?.x ?? 0), (nodes[a]?.y ?? 0) - (hub?.y ?? 0)),
                Math.hypot((nodes[b]?.x ?? 0) - (hub?.x ?? 0), (nodes[b]?.y ?? 0) - (hub?.y ?? 0)),
            );

        // The silk at rest: faint, so the field stays a field. While the web is
        // being spun, a thread appears once the spinning front has passed it.
        // Radials are the strong threads and drawn so; the spiral is finer.
        for (const radial of [true, false]) {
            ctx.lineWidth = radial ? 1 : 0.7;
            ctx.strokeStyle = radial ? 'rgba(159, 214, 236, 0.2)' : 'rgba(159, 214, 236, 0.13)';
            ctx.beginPath();
            for (const t of threads) {
                if (t.radial !== radial) continue;
                if (spun < 1 && reachOf(t.a, t.b) > radius) continue;
                ctx.moveTo(px(t.a), py(t.a));
                ctx.lineTo(px(t.b), py(t.b));
            }
            ctx.stroke();
        }
        // Everything lit from here on adds light rather than paint.
        ctx.globalCompositeOperation = 'lighter';
        if (spun >= 1) {
            for (const m of motes) {
                const t = threads[m.thread];
                if (!t) continue;
                const life = (now - m.born) / m.life;
                const a = Math.sin(Math.min(1, Math.max(0, life)) * Math.PI) * 0.85;
                const x = px(t.a) + (px(t.b) - px(t.a)) * m.t;
                const y = py(t.a) + (py(t.b) - py(t.a)) * m.t;
                ctx.fillStyle = `rgba(159, 214, 236, ${(a * 0.18).toFixed(3)})`;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = `rgba(232, 246, 252, ${(a * 0.7).toFixed(3)})`;
                ctx.beginPath();
                ctx.arc(x, y, 1.3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        // The spinning front: a bright edge running outward on the first pass.
        if (spun < 1 && hub) {
            ctx.strokeStyle = `rgba(214, 238, 248, ${(0.5 * (1 - spun)).toFixed(3)})`;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            for (const t of threads) {
                const r = reachOf(t.a, t.b);
                if (r > radius || r < radius - 60) continue;
                ctx.moveTo(px(t.a), py(t.a));
                ctx.lineTo(px(t.b), py(t.b));
            }
            ctx.stroke();
        }

        for (const pulse of pulses) {
            // The ring a seed throws where it lands.
            const seedId = [...pulse.lit.entries()].find(([, l]) => l.from < 0)?.[0];
            const age = (now - pulse.started) / RIPPLE_MS;
            if (seedId !== undefined && age < 1) {
                ctx.strokeStyle = `rgba(214, 238, 248, ${((1 - age) * 0.35).toFixed(3)})`;
                ctx.lineWidth = 1.5 * (1 - age) + 0.4;
                ctx.beginPath();
                ctx.arc(px(seedId), py(seedId), 6 + age * 90, 0, Math.PI * 2);
                ctx.stroke();
            }
            // The pulse: a thread lights from the node that fed it to the node it
            // fed, growing along its length while the hop is in flight.
            for (const [id, l] of pulse.lit) {
                if (l.from < 0 || now < l.at - HOP_MS) continue;
                const travel = Math.min(1, (now - (l.at - HOP_MS)) / HOP_MS);
                const alpha = Math.max(
                    glow(pulse, l.from, now),
                    travel < 1 ? 0.7 : glow(pulse, id, now),
                );
                if (alpha <= 0.01) continue;
                const ax = px(l.from);
                const ay = py(l.from);
                const bx = ax + (px(id) - ax) * travel;
                const by = ay + (py(id) - ay) * travel;
                // A wide soft stroke under a fine bright one: silk catching light.
                ctx.strokeStyle = `rgba(110, 190, 230, ${(alpha * 0.16).toFixed(3)})`;
                ctx.lineWidth = 3 + alpha * 4;
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
                ctx.strokeStyle = `rgba(225, 244, 252, ${Math.min(0.62, alpha * 0.65).toFixed(3)})`;
                ctx.lineWidth = 0.9 + alpha * 1.6;
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
                // The spark at the head of a hop still in flight.
                if (travel < 1) {
                    ctx.fillStyle = 'rgba(159, 214, 236, 0.15)';
                    ctx.beginPath();
                    ctx.arc(bx, by, 7, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = 'rgba(245, 252, 255, 0.65)';
                    ctx.beginPath();
                    ctx.arc(bx, by, 2.4, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            for (const [id] of pulse.lit) {
                const g = glow(pulse, id, now);
                if (g <= 0.01) continue;
                // A soft halo, then the bright core.
                ctx.fillStyle = `rgba(159, 214, 236, ${(g * 0.1).toFixed(3)})`;
                ctx.beginPath();
                ctx.arc(px(id), py(id), 5 + g * 11, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = `rgba(232, 246, 252, ${(g * 0.6).toFixed(3)})`;
                ctx.beginPath();
                ctx.arc(px(id), py(id), 1.6 + g * 3.4, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalCompositeOperation = 'source-over';
    };

    const tick = (now: number): void => {
        frame = 0;
        if (!visible || paused) return;
        const dt = Math.min(64, now - lastFrame);
        lastFrame = now;
        settle(now);
        drift(now, dt);
        if (!firstSeed && now - spunAt > SPIN_MS * 0.5) {
            // The first query lands on the hub once the web is mostly spun.
            firstSeed = true;
            seed(0, now);
        } else if (firstSeed && now - lastSeed > AUTO_MS) {
            // Anywhere on the inner two-thirds of the web, so the field is never
            // quiet for long and no corner is favoured.
            seed(1 + Math.floor(Math.random() * RADIALS * Math.round(RINGS * 0.66)), now);
        }
        draw(now);
        frame = requestAnimationFrame(tick);
    };
    const start = (): void => {
        lastFrame = performance.now();
        if (!frame && !reduced) frame = requestAnimationFrame(tick);
    };
    const stop = (): void => {
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
    };

    const still = (): void => {
        // One finished pulse, held: the web as a reduced-motion reader sees it.
        const now = performance.now();
        seed(0, now - HOP_MS * 8);
        const pulse = pulses[pulses.length - 1];
        if (pulse) for (const l of pulse.lit.values()) l.at = now - FADE_MS * 0.35;
        draw(now);
    };

    build();
    if (reduced) {
        spunAt = -SPIN_MS;
        still();
        if (toggle) toggle.hidden = true;
    } else {
        spunAt = performance.now();
        start();
    }

    new ResizeObserver(() => {
        build();
        if (reduced) still();
        else draw(performance.now());
        // A resize rebuilds the web at rest; it is not spun again.
        spunAt = Math.min(spunAt, performance.now() - SPIN_MS);
    }).observe(field);

    new IntersectionObserver((entries) => {
        visible = entries.some((e) => e.isIntersecting);
        if (visible) start();
        else stop();
    }).observe(field);

    document.addEventListener('visibilitychange', () => {
        visible = document.visibilityState === 'visible';
        if (visible) start();
        else stop();
    });

    field.addEventListener('click', (event) => {
        if (reduced || paused) return;
        // Only the field itself: links, tabs and copy buttons keep their clicks.
        if ((event.target as Element).closest('a, button, input, [role="tab"], pre, code')) return;
        const box = field.getBoundingClientRect();
        seed(nearest(event.clientX - box.left, event.clientY - box.top), performance.now());
        start();
    });

    toggle?.addEventListener('click', () => {
        paused = !paused;
        toggle.setAttribute('aria-pressed', String(paused));
        toggle.textContent = paused ? 'play the web' : 'pause the web';
        if (paused) stop();
        else start();
    });
}
