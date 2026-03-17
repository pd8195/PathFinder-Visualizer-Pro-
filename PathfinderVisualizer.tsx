import { useState, useEffect, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
type NodeType = "empty" | "wall" | "start" | "end" | "weight";
type AlgoName = "bfs" | "dfs" | "dijkstra" | "astar" | "bidibfs";
type VisitedMap = Map<string, string>; // key -> parentKey

interface GridNode {
  row: number;
  col: number;
  type: NodeType;
  weight: number;
  visitedBy: AlgoName[];
  inPath: AlgoName[];
  gScore?: number;
  fScore?: number;
}

interface AlgoResult {
  visited: [number, number][];
  path: [number, number][];
  cost: number;
  nodesExplored: number;
}

interface AlgoState {
  name: AlgoName;
  label: string;
  color: string;
  glow: string;
  result: AlgoResult | null;
  running: boolean;
  done: boolean;
  visitedCount: number;
  pathLength: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────
const ROWS = 22;
const COLS = 48;
const CELL = 22;

const ALGO_CONFIG: Record<AlgoName, { label: string; color: string; glow: string; desc: string }> = {
  bfs:      { label: "BFS",       color: "#00f5ff", glow: "#00f5ff80", desc: "Breadth-First Search — guarantees shortest path (unweighted)" },
  dfs:      { label: "DFS",       color: "#ff6b35", glow: "#ff6b3580", desc: "Depth-First Search — fast but NOT guaranteed shortest" },
  dijkstra: { label: "Dijkstra",  color: "#a855f7", glow: "#a855f780", desc: "Dijkstra — shortest path with weights" },
  astar:    { label: "A*",        color: "#22c55e", glow: "#22c55e80", desc: "A* Search — heuristic-guided, fastest to goal" },
  bidibfs:  { label: "Bi-BFS",    color: "#f59e0b", glow: "#f59e0b80", desc: "Bidirectional BFS — searches from both ends" },
};

const WEIGHT_VALS = [1, 3, 5, 10];
const WEIGHT_COLORS = ["", "#fde68a", "#fbbf24", "#ef4444"];

const key = (r: number, c: number) => `${r},${c}`;
const unkey = (k: string): [number, number] => k.split(",").map(Number) as [number, number];

// ─── Algorithm Implementations ────────────────────────────────────────────────
function getNeighbors(r: number, c: number, grid: GridNode[][]): GridNode[] {
  const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
  return dirs
    .map(([dr, dc]) => grid[r + dr]?.[c + dc])
    .filter((n): n is GridNode => !!n && n.type !== "wall");
}

function reconstructPath(parent: VisitedMap, endKey: string): [number, number][] {
  const path: [number, number][] = [];
  let cur = endKey;
  while (cur) {
    path.unshift(unkey(cur));
    cur = parent.get(cur)!;
  }
  return path;
}

function runBFS(grid: GridNode[][], start: [number,number], end: [number,number]): AlgoResult {
  const visited: [number,number][] = [];
  const parent: VisitedMap = new Map();
  const seen = new Set<string>();
  const queue: [number,number][] = [start];
  const startKey = key(...start);
  const endKey = key(...end);
  seen.add(startKey);
  parent.set(startKey, "");

  while (queue.length) {
    const [r, c] = queue.shift()!;
    const k = key(r, c);
    if (k !== startKey) visited.push([r, c]);
    if (k === endKey) {
      const path = reconstructPath(parent, endKey);
      return { visited, path, cost: path.length - 1, nodesExplored: visited.length };
    }
    for (const nb of getNeighbors(r, c, grid)) {
      const nk = key(nb.row, nb.col);
      if (!seen.has(nk)) {
        seen.add(nk);
        parent.set(nk, k);
        queue.push([nb.row, nb.col]);
      }
    }
  }
  return { visited, path: [], cost: 0, nodesExplored: visited.length };
}

function runDFS(grid: GridNode[][], start: [number,number], end: [number,number]): AlgoResult {
  const visited: [number,number][] = [];
  const parent: VisitedMap = new Map();
  const seen = new Set<string>();
  const stack: [number,number][] = [start];
  const startKey = key(...start);
  const endKey = key(...end);
  parent.set(startKey, "");

  while (stack.length) {
    const [r, c] = stack.pop()!;
    const k = key(r, c);
    if (seen.has(k)) continue;
    seen.add(k);
    if (k !== startKey) visited.push([r, c]);
    if (k === endKey) {
      const path = reconstructPath(parent, endKey);
      return { visited, path, cost: path.length - 1, nodesExplored: visited.length };
    }
    for (const nb of getNeighbors(r, c, grid)) {
      const nk = key(nb.row, nb.col);
      if (!seen.has(nk)) {
        if (!parent.has(nk)) parent.set(nk, k);
        stack.push([nb.row, nb.col]);
      }
    }
  }
  return { visited, path: [], cost: 0, nodesExplored: visited.length };
}

class MinHeap<T> {
  private data: { priority: number; item: T }[] = [];
  push(item: T, priority: number) {
    this.data.push({ priority, item });
    this.data.sort((a, b) => a.priority - b.priority);
  }
  pop(): T | undefined { return this.data.shift()?.item; }
  get size() { return this.data.length; }
}

function runDijkstra(grid: GridNode[][], start: [number,number], end: [number,number]): AlgoResult {
  const visited: [number,number][] = [];
  const parent: VisitedMap = new Map();
  const dist = new Map<string, number>();
  const heap = new MinHeap<[number,number]>();
  const startKey = key(...start);
  const endKey = key(...end);
  dist.set(startKey, 0);
  parent.set(startKey, "");
  heap.push(start, 0);

  while (heap.size) {
    const [r, c] = heap.pop()!;
    const k = key(r, c);
    if (k !== startKey) visited.push([r, c]);
    if (k === endKey) {
      const path = reconstructPath(parent, endKey);
      return { visited, path, cost: dist.get(endKey) ?? 0, nodesExplored: visited.length };
    }
    for (const nb of getNeighbors(r, c, grid)) {
      const nk = key(nb.row, nb.col);
      const newDist = (dist.get(k) ?? Infinity) + nb.weight;
      if (newDist < (dist.get(nk) ?? Infinity)) {
        dist.set(nk, newDist);
        parent.set(nk, k);
        heap.push([nb.row, nb.col], newDist);
      }
    }
  }
  return { visited, path: [], cost: 0, nodesExplored: visited.length };
}

function manhattan(a: [number,number], b: [number,number]) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

function runAStar(grid: GridNode[][], start: [number,number], end: [number,number]): AlgoResult {
  const visited: [number,number][] = [];
  const parent: VisitedMap = new Map();
  const gScore = new Map<string, number>();
  const heap = new MinHeap<[number,number]>();
  const startKey = key(...start);
  const endKey = key(...end);
  gScore.set(startKey, 0);
  parent.set(startKey, "");
  heap.push(start, manhattan(start, end));

  while (heap.size) {
    const [r, c] = heap.pop()!;
    const k = key(r, c);
    if (k !== startKey) visited.push([r, c]);
    if (k === endKey) {
      const path = reconstructPath(parent, endKey);
      return { visited, path, cost: gScore.get(endKey) ?? 0, nodesExplored: visited.length };
    }
    for (const nb of getNeighbors(r, c, grid)) {
      const nk = key(nb.row, nb.col);
      const tentG = (gScore.get(k) ?? Infinity) + nb.weight;
      if (tentG < (gScore.get(nk) ?? Infinity)) {
        gScore.set(nk, tentG);
        parent.set(nk, k);
        heap.push([nb.row, nb.col], tentG + manhattan([nb.row, nb.col], end));
      }
    }
  }
  return { visited, path: [], cost: 0, nodesExplored: visited.length };
}

function runBidiBFS(grid: GridNode[][], start: [number,number], end: [number,number]): AlgoResult {
  const visited: [number,number][] = [];
  const frontierS = new Set<string>([key(...start)]);
  const frontierE = new Set<string>([key(...end)]);
  const parentS: VisitedMap = new Map([[key(...start), ""]]);
  const parentE: VisitedMap = new Map([[key(...end), ""]]);
  const seenS = new Set<string>([key(...start)]);
  const seenE = new Set<string>([key(...end)]);

  while (frontierS.size && frontierE.size) {
    const nextS = new Set<string>();
    for (const k of frontierS) {
      const [r, c] = unkey(k);
      for (const nb of getNeighbors(r, c, grid)) {
        const nk = key(nb.row, nb.col);
        if (!seenS.has(nk)) {
          seenS.add(nk);
          parentS.set(nk, k);
          nextS.add(nk);
          visited.push([nb.row, nb.col]);
          if (seenE.has(nk)) {
            const pathS = reconstructPath(parentS, nk);
            const pathE = reconstructPath(parentE, nk).reverse().slice(1);
            const path = [...pathS, ...pathE];
            return { visited, path, cost: path.length - 1, nodesExplored: visited.length };
          }
        }
      }
    }
    frontierS.clear(); nextS.forEach(k => frontierS.add(k));

    const nextE = new Set<string>();
    for (const k of frontierE) {
      const [r, c] = unkey(k);
      for (const nb of getNeighbors(r, c, grid)) {
        const nk = key(nb.row, nb.col);
        if (!seenE.has(nk)) {
          seenE.add(nk);
          parentE.set(nk, k);
          nextE.add(nk);
          visited.push([nb.row, nb.col]);
          if (seenS.has(nk)) {
            const pathS = reconstructPath(parentS, nk);
            const pathE = reconstructPath(parentE, nk).reverse().slice(1);
            const path = [...pathS, ...pathE];
            return { visited, path, cost: path.length - 1, nodesExplored: visited.length };
          }
        }
      }
    }
    frontierE.clear(); nextE.forEach(k => frontierE.add(k));
  }
  return { visited, path: [], cost: 0, nodesExplored: visited.length };
}

// ─── Maze Generators ──────────────────────────────────────────────────────────
function generateRecursiveMaze(rows: number, cols: number): Set<string> {
  const walls = new Set<string>();
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (r % 2 === 0 || c % 2 === 0) walls.add(key(r, c));

  function carve(r: number, c: number) {
    const dirs = [[-2,0],[2,0],[0,-2],[0,2]].sort(() => Math.random() - 0.5);
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && walls.has(key(nr, nc))) {
        walls.delete(key(nr, nc));
        walls.delete(key(r + dr/2, c + dc/2));
        carve(nr, nc);
      }
    }
  }
  walls.delete(key(1, 1));
  carve(1, 1);
  return walls;
}

// ─── Main Component ───────────────────────────────────────────────────────────
const makeGrid = (): GridNode[][] =>
  Array.from({ length: ROWS }, (_, r) =>
    Array.from({ length: COLS }, (_, c) => ({
      row: r, col: c, type: "empty" as NodeType, weight: 1,
      visitedBy: [], inPath: [],
    }))
  );

const ALGOS: AlgoName[] = ["bfs", "dfs", "dijkstra", "astar", "bidibfs"];

export default function PathfinderVisualizer() {
  const [grid, setGrid] = useState<GridNode[][]>(() => {
    const g = makeGrid();
    g[10][5].type = "start";
    g[10][42].type = "end";
    return g;
  });

  const [start, setStart] = useState<[number,number]>([10, 5]);
  const [end, setEnd] = useState<[number,number]>([10, 42]);
  const [tool, setTool] = useState<"wall" | "weight" | "eraser" | "start" | "end">("wall");
  const [weightVal, setWeightVal] = useState(3);
  const [selectedAlgos, setSelectedAlgos] = useState<Set<AlgoName>>(new Set(ALGOS));
  const [algoStates, setAlgoStates] = useState<Record<AlgoName, AlgoState>>(() =>
    Object.fromEntries(ALGOS.map(a => [a, {
      name: a, ...ALGO_CONFIG[a], result: null,
      running: false, done: false, visitedCount: 0, pathLength: 0,
    }])) as Record<AlgoName, AlgoState>
  );
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [mouseDown, setMouseDown] = useState(false);
  const [speed, setSpeed] = useState(12);
  const [showLegend, setShowLegend] = useState(true);
  const animRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const gridRef = useRef(grid);
  gridRef.current = grid;

  const clearAnimations = () => {
    animRef.current.forEach(clearTimeout);
    animRef.current = [];
  };

  const resetVisualization = useCallback(() => {
    clearAnimations();
    setIsRunning(false);
    setIsDone(false);
    setAlgoStates(prev => Object.fromEntries(
      ALGOS.map(a => [a, { ...prev[a], result: null, running: false, done: false, visitedCount: 0, pathLength: 0 }])
    ) as Record<AlgoName, AlgoState>);
    setGrid(g => g.map(row => row.map(cell => ({ ...cell, visitedBy: [], inPath: [] }))));
  }, []);

  const clearAll = useCallback(() => {
    clearAnimations();
    setIsRunning(false);
    setIsDone(false);
    setAlgoStates(prev => Object.fromEntries(
      ALGOS.map(a => [a, { ...prev[a], result: null, running: false, done: false, visitedCount: 0, pathLength: 0 }])
    ) as Record<AlgoName, AlgoState>);
    const g = makeGrid();
    g[start[0]][start[1]].type = "start";
    g[end[0]][end[1]].type = "end";
    setGrid(g);
  }, [start, end]);

  const generateMaze = useCallback(() => {
    resetVisualization();
    const walls = generateRecursiveMaze(ROWS, COLS);
    setGrid(g => g.map(row => row.map(cell => {
      const k = key(cell.row, cell.col);
      const isStart = cell.row === start[0] && cell.col === start[1];
      const isEnd = cell.row === end[0] && cell.col === end[1];
      return {
        ...cell, visitedBy: [], inPath: [],
        type: isStart ? "start" : isEnd ? "end" : walls.has(k) ? "wall" : "empty",
        weight: 1,
      };
    })));
  }, [start, end, resetVisualization]);

  const handleCellInteract = useCallback((r: number, c: number) => {
    if (isRunning) return;
    setGrid(g => {
      const ng = g.map(row => [...row.map(cell => ({ ...cell }))]);
      const cell = ng[r][c];
      if (tool === "start") {
        ng[start[0]][start[1]].type = "empty";
        cell.type = "start";
        setStart([r, c]);
      } else if (tool === "end") {
        ng[end[0]][end[1]].type = "empty";
        cell.type = "end";
        setEnd([r, c]);
      } else if (cell.type === "start" || cell.type === "end") {
        // don't overwrite
      } else if (tool === "wall") {
        cell.type = cell.type === "wall" ? "empty" : "wall";
        cell.weight = 1;
      } else if (tool === "weight") {
        cell.type = "weight";
        cell.weight = weightVal;
      } else if (tool === "eraser") {
        cell.type = "empty";
        cell.weight = 1;
      }
      return ng;
    });
  }, [tool, start, end, weightVal, isRunning]);

  const runVisualization = useCallback(() => {
    resetVisualization();
    const g = gridRef.current;
    const active = ALGOS.filter(a => selectedAlgos.has(a));
    if (!active.length) return;

    setIsRunning(true);
    setAlgoStates(prev => {
      const next = { ...prev };
      active.forEach(a => { next[a] = { ...next[a], running: true, done: false }; });
      return next;
    });

    const results: Record<AlgoName, AlgoResult> = {} as any;
    active.forEach(a => {
      const fn = { bfs: runBFS, dfs: runDFS, dijkstra: runDijkstra, astar: runAStar, bidibfs: runBidiBFS }[a];
      results[a] = fn(g, start, end);
    });

    // interleave visited animations
    const maxVisited = Math.max(...active.map(a => results[a].visited.length));
    const delay = Math.max(2, 30 - speed * 2);

    let t = 0;
    for (let i = 0; i < maxVisited; i++) {
      const frame = i;
      const tid = setTimeout(() => {
        setGrid(prev => {
          const ng = prev.map(row => [...row.map(c => ({ ...c }))]);
          active.forEach(a => {
            if (frame < results[a].visited.length) {
              const [r, c] = results[a].visited[frame];
              if (!ng[r][c].visitedBy.includes(a))
                ng[r][c].visitedBy = [...ng[r][c].visitedBy, a];
            }
          });
          return ng;
        });
        setAlgoStates(prev => {
          const next = { ...prev };
          active.forEach(a => {
            next[a] = { ...next[a], visitedCount: Math.min(frame + 1, results[a].visited.length) };
          });
          return next;
        });
      }, frame * delay);
      animRef.current.push(tid);
      t = frame * delay;
    }

    // then show paths
    const maxPath = Math.max(...active.map(a => results[a].path.length));
    for (let i = 0; i < maxPath; i++) {
      const frame = i;
      const tid = setTimeout(() => {
        setGrid(prev => {
          const ng = prev.map(row => [...row.map(c => ({ ...c }))]);
          active.forEach(a => {
            if (frame < results[a].path.length) {
              const [r, c] = results[a].path[frame];
              if (!ng[r][c].inPath.includes(a))
                ng[r][c].inPath = [...ng[r][c].inPath, a];
            }
          });
          return ng;
        });
      }, t + 50 + frame * 20);
      animRef.current.push(tid);
    }

    const finalTid = setTimeout(() => {
      setIsRunning(false);
      setIsDone(true);
      setAlgoStates(prev => {
        const next = { ...prev };
        active.forEach(a => {
          next[a] = { ...next[a], running: false, done: true, result: results[a], pathLength: results[a].path.length };
        });
        return next;
      });
    }, t + 50 + maxPath * 20 + 100);
    animRef.current.push(finalTid);
  }, [start, end, selectedAlgos, speed, resetVisualization]);

  const getCellStyle = (cell: GridNode): React.CSSProperties => {
    const base: React.CSSProperties = {
      width: CELL, height: CELL,
      display: "flex", alignItems: "center", justifyContent: "center",
      cursor: "crosshair", userSelect: "none",
      transition: "background 0.15s, box-shadow 0.15s",
      position: "relative", fontSize: 10, fontWeight: 700,
      borderRadius: 2,
    };

    if (cell.type === "start") return { ...base, background: "#00ff88", boxShadow: "0 0 12px #00ff88", zIndex: 2, borderRadius: 4 };
    if (cell.type === "end") return { ...base, background: "#ff3366", boxShadow: "0 0 12px #ff3366", zIndex: 2, borderRadius: 4 };
    if (cell.type === "wall") return { ...base, background: "#1e1b4b", boxShadow: "inset 0 0 8px #0008", cursor: "crosshair" };

    if (cell.inPath.length > 0) {
      const colors = cell.inPath.map(a => ALGO_CONFIG[a].color);
      if (colors.length === 1) return { ...base, background: colors[0], boxShadow: `0 0 10px ${colors[0]}`, zIndex: 1 };
      const grad = `conic-gradient(${colors.map((c, i) => `${c} ${i * 360/colors.length}deg ${(i+1)*360/colors.length}deg`).join(",")})`;
      return { ...base, background: grad, zIndex: 1 };
    }

    if (cell.visitedBy.length > 0) {
      const colors = cell.visitedBy.map(a => ALGO_CONFIG[a].color + "55");
      if (colors.length === 1) {
        const color = ALGO_CONFIG[cell.visitedBy[0]].color;
        return { ...base, background: color + "33", borderBottom: `1px solid ${color}44` };
      }
      return { ...base, background: `linear-gradient(135deg,${colors.join(",")})` };
    }

    if (cell.type === "weight") {
      const wi = WEIGHT_VALS.indexOf(cell.weight);
      return { ...base, background: WEIGHT_COLORS[wi] || "#fbbf24", color: "#78350f" };
    }

    return { ...base, background: "transparent" };
  };

  const toggleAlgo = (a: AlgoName) => {
    setSelectedAlgos(prev => {
      const next = new Set(prev);
      if (next.has(a)) { if (next.size > 1) next.delete(a); }
      else next.add(a);
      return next;
    });
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#050510",
      fontFamily: "'Orbitron', 'Courier New', monospace",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "16px 8px", gap: 12,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { overflow-x: hidden; }
        .grid-cell:hover { filter: brightness(1.4); }
        .btn { transition: all 0.2s; }
        .btn:hover { transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        @keyframes glow { 0%,100%{box-shadow:0 0 8px currentColor} 50%{box-shadow:0 0 24px currentColor} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }
        .running-badge { animation: pulse 1s infinite; }
        .stat-card { animation: fadeIn 0.3s ease; }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: "center", animation: "fadeIn 0.5s ease" }}>
        <h1 style={{
          fontSize: 28, fontWeight: 900, letterSpacing: 4,
          background: "linear-gradient(90deg, #00f5ff, #a855f7, #ff6b35, #22c55e)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          textShadow: "none", marginBottom: 2,
        }}>⬡ PATHFINDER VISUALIZER PRO</h1>
        <p style={{ color: "#6366f1", fontSize: 11, letterSpacing: 3 }}>ALGORITHM RACING ENGINE v1.0</p>
      </div>

      {/* Controls Bar */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center",
        justifyContent: "center", maxWidth: 1200,
      }}>
        {/* Tools */}
        <div style={{ display: "flex", gap: 4, background: "#0f0f2a", borderRadius: 8, padding: 4, border: "1px solid #1e1b4b" }}>
          {([
            ["wall",    "🧱", "#6366f1"],
            ["weight",  "⚖️",  "#f59e0b"],
            ["eraser",  "✏️",  "#64748b"],
            ["start",   "🟢", "#22c55e"],
            ["end",     "🔴", "#ef4444"],
          ] as [string, string, string][]).map(([t, icon, col]) => (
            <button key={t} className="btn" onClick={() => setTool(t as any)} style={{
              background: tool === t ? col + "33" : "transparent",
              border: `1px solid ${tool === t ? col : "#1e1b4b"}`,
              color: tool === t ? col : "#64748b",
              padding: "4px 10px", borderRadius: 6, cursor: "pointer",
              fontSize: 11, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1,
              boxShadow: tool === t ? `0 0 8px ${col}66` : "none",
            }}>{icon} {t.toUpperCase()}</button>
          ))}
        </div>

        {/* Weight selector */}
        {tool === "weight" && (
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ color: "#64748b", fontSize: 10 }}>W:</span>
            {WEIGHT_VALS.filter(v => v > 1).map((v, i) => (
              <button key={v} className="btn" onClick={() => setWeightVal(v)} style={{
                background: weightVal === v ? WEIGHT_COLORS[i+1] : "#0f0f2a",
                border: `1px solid ${WEIGHT_COLORS[i+1]}`,
                color: weightVal === v ? "#78350f" : WEIGHT_COLORS[i+1],
                padding: "3px 8px", borderRadius: 4, cursor: "pointer",
                fontSize: 11, fontFamily: "inherit", fontWeight: 700,
              }}>{v}</button>
            ))}
          </div>
        )}

        {/* Speed */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ color: "#6366f1", fontSize: 10, letterSpacing: 1 }}>⚡ SPEED</span>
          <input type="range" min={1} max={14} value={speed} onChange={e => setSpeed(+e.target.value)}
            style={{ width: 80, accentColor: "#6366f1" }} />
          <span style={{ color: "#a855f7", fontSize: 10, minWidth: 20 }}>{speed}</span>
        </div>

        {/* Action buttons */}
        <button className="btn" onClick={generateMaze} style={{
          background: "#0f0f2a", border: "1px solid #6366f1", color: "#6366f1",
          padding: "5px 14px", borderRadius: 6, cursor: "pointer",
          fontSize: 11, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1,
        }}>🌀 MAZE</button>

        <button className="btn" onClick={clearAll} style={{
          background: "#0f0f2a", border: "1px solid #ef4444", color: "#ef4444",
          padding: "5px 14px", borderRadius: 6, cursor: "pointer",
          fontSize: 11, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1,
        }}>⬜ CLEAR</button>

        <button className="btn" onClick={resetVisualization} style={{
          background: "#0f0f2a", border: "1px solid #f59e0b", color: "#f59e0b",
          padding: "5px 14px", borderRadius: 6, cursor: "pointer",
          fontSize: 11, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1,
        }}>↺ RESET VIZ</button>

        <button className="btn" onClick={isRunning ? clearAnimations : runVisualization} style={{
          background: isRunning ? "#7f1d1d" : "linear-gradient(135deg, #6366f1, #a855f7)",
          border: "none", color: "#fff",
          padding: "5px 20px", borderRadius: 6, cursor: "pointer",
          fontSize: 12, fontFamily: "inherit", fontWeight: 900, letterSpacing: 2,
          boxShadow: isRunning ? "0 0 12px #ef444466" : "0 0 16px #6366f166",
        }}>{isRunning ? "⏹ STOP" : "▶ RACE!"}</button>
      </div>

      {/* Algorithm selector */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center" }}>
        {ALGOS.map(a => {
          const cfg = ALGO_CONFIG[a];
          const st = algoStates[a];
          const sel = selectedAlgos.has(a);
          return (
            <button key={a} className="btn" onClick={() => toggleAlgo(a)} title={cfg.desc} style={{
              background: sel ? cfg.color + "22" : "#0a0a1a",
              border: `1.5px solid ${sel ? cfg.color : "#1e1b4b"}`,
              color: sel ? cfg.color : "#334155",
              padding: "4px 12px", borderRadius: 20, cursor: "pointer",
              fontSize: 11, fontFamily: "inherit", fontWeight: 700, letterSpacing: 1,
              boxShadow: sel ? `0 0 8px ${cfg.glow}` : "none",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: sel ? cfg.color : "#334155",
                boxShadow: sel && st.running ? `0 0 8px ${cfg.color}` : "none",
                animation: st.running ? "pulse 0.8s infinite" : "none",
              }} />
              {cfg.label}
              {st.done && st.result && (
                <span style={{ fontSize: 9, opacity: 0.8 }}>
                  {st.result.path.length ? `${st.result.path.length - 1}` : "✗"}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${COLS}, ${CELL}px)`,
          gridTemplateRows: `repeat(${ROWS}, ${CELL}px)`,
          gap: 1,
          background: "#0d0d1f",
          padding: 4,
          borderRadius: 8,
          border: "1px solid #1e1b4b",
          boxShadow: "0 0 40px #6366f122, inset 0 0 30px #0005",
          cursor: "crosshair",
        }}
        onMouseLeave={() => setMouseDown(false)}
      >
        {grid.flat().map(cell => (
          <div
            key={`${cell.row}-${cell.col}`}
            className="grid-cell"
            style={getCellStyle(cell)}
            onMouseDown={() => { setMouseDown(true); handleCellInteract(cell.row, cell.col); }}
            onMouseEnter={() => mouseDown && handleCellInteract(cell.row, cell.col)}
            onMouseUp={() => setMouseDown(false)}
          >
            {cell.type === "start" && <span style={{ fontSize: 13 }}>▶</span>}
            {cell.type === "end" && <span style={{ fontSize: 13 }}>✦</span>}
            {cell.type === "weight" && cell.weight > 1 && (
              <span style={{ fontSize: 8, color: "#92400e" }}>{cell.weight}</span>
            )}
          </div>
        ))}
      </div>

      {/* Stats Panel */}
      {(isRunning || isDone) && (
        <div style={{
          display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center",
          maxWidth: 1200,
        }}>
          {ALGOS.filter(a => selectedAlgos.has(a)).map(a => {
            const cfg = ALGO_CONFIG[a];
            const st = algoStates[a];
            const res = st.result;
            return (
              <div key={a} className="stat-card" style={{
                background: "#0a0a1a",
                border: `1px solid ${cfg.color}44`,
                borderRadius: 10, padding: "10px 16px",
                minWidth: 140,
                boxShadow: st.running ? `0 0 16px ${cfg.glow}` : "none",
              }}>
                <div style={{ color: cfg.color, fontSize: 12, fontWeight: 900, letterSpacing: 2, marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: "50%", background: cfg.color,
                    animation: st.running ? "pulse 0.8s infinite" : "none",
                    display: "inline-block",
                  }} />
                  {cfg.label}
                  {st.running && <span className="running-badge" style={{ fontSize: 9, color: "#94a3b8" }}>RUNNING</span>}
                  {st.done && !res?.path.length && <span style={{ fontSize: 9, color: "#ef4444" }}>NO PATH</span>}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <span style={{ color: "#64748b", fontSize: 9 }}>EXPLORED</span>
                    <span style={{ color: "#e2e8f0", fontSize: 10, fontWeight: 700 }}>{st.visitedCount}</span>
                  </div>
                  {st.done && res && (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                        <span style={{ color: "#64748b", fontSize: 9 }}>PATH LEN</span>
                        <span style={{ color: cfg.color, fontSize: 10, fontWeight: 700 }}>
                          {res.path.length ? res.path.length - 1 : "—"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                        <span style={{ color: "#64748b", fontSize: 9 }}>COST</span>
                        <span style={{ color: cfg.color, fontSize: 10, fontWeight: 700 }}>
                          {res.path.length ? res.cost : "—"}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Legend */}
      {showLegend && (
        <div style={{
          display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center",
          fontSize: 9, color: "#64748b", letterSpacing: 1, padding: "4px 8px",
          borderTop: "1px solid #1e1b4b", paddingTop: 8,
        }}>
          {[
            ["▶", "#00ff88", "START"],
            ["✦", "#ff3366", "END"],
            ["  ", "#1e1b4b", "WALL"],
            ["3", "#fbbf24", "WEIGHT"],
            ...ALGOS.map(a => ["  ", ALGO_CONFIG[a].color + "44", ALGO_CONFIG[a].label + " VISITED"]),
            ...ALGOS.map(a => ["  ", ALGO_CONFIG[a].color, ALGO_CONFIG[a].label + " PATH"]),
          ].map(([icon, color, label]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{
                width: 14, height: 14, borderRadius: 2,
                background: color, display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 8, color: "#000",
              }}>{icon}</div>
              <span>{label}</span>
            </div>
          ))}
          <button onClick={() => setShowLegend(false)} style={{
            background: "none", border: "none", color: "#334155", cursor: "pointer", fontSize: 9
          }}>✕ hide</button>
        </div>
      )}

      <div style={{ color: "#1e293b", fontSize: 9, letterSpacing: 2 }}>
        CLICK/DRAG TO DRAW • SELECT ALGORITHMS ABOVE • HIT RACE TO VISUALIZE
      </div>
    </div>
  );
}