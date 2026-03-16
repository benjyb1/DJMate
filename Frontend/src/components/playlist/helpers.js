// playlist/helpers.js — tree-building utilities for PlaylistOrganiser

/**
 * Build a hierarchical tree from a flat list of playlists (DB records with parent_id).
 */
export function buildTree(flatList) {
  const map = {};
  const roots = [];
  for (const item of flatList) {
    map[item.id] = { ...item, children: [] };
  }
  for (const item of flatList) {
    if (item.parent_id && map[item.parent_id]) {
      map[item.parent_id].children.push(map[item.id]);
    } else {
      roots.push(map[item.id]);
    }
  }
  return roots;
}

/**
 * Build a virtual folder tree from track filepaths.
 * Returns { roots: [node], tracksByPath: { "/full/path": [track, ...] } }
 * Each node: { id: path, name, children: [], trackCount }
 */
export function buildFileTree(tracks) {
  const dirMap = {};      // path → { id, name, children: Set, trackCount }
  const tracksByPath = {}; // path → [track, ...]

  // Find common prefix to strip (e.g. /Users/benjyb/Music/)
  const allDirs = [];
  for (const t of tracks) {
    if (!t.filepath) continue;
    const parts = t.filepath.split('/');
    parts.pop(); // remove filename
    const dir = parts.join('/');
    if (dir) allDirs.push(dir);
  }
  // Find longest common prefix
  let prefix = '';
  if (allDirs.length > 0) {
    const sorted = allDirs.slice().sort();
    const first = sorted[0].split('/');
    const last = sorted[sorted.length - 1].split('/');
    const common = [];
    for (let i = 0; i < first.length && i < last.length; i++) {
      if (first[i] === last[i]) common.push(first[i]);
      else break;
    }
    prefix = common.join('/');
  }

  for (const t of tracks) {
    if (!t.filepath) continue;
    const parts = t.filepath.split('/');
    parts.pop();
    const dir = parts.join('/');

    // Add track to its directory
    if (!tracksByPath[dir]) tracksByPath[dir] = [];
    tracksByPath[dir].push(t);

    // Build all ancestor directories
    for (let i = 1; i <= parts.length; i++) {
      const path = parts.slice(0, i).join('/');
      if (!dirMap[path]) {
        dirMap[path] = { id: 'lib:' + path, path, name: parts[i - 1], childPaths: new Set(), trackCount: 0 };
      }
    }
    dirMap[dir].trackCount++;

    // Register parent → child relationships
    for (let i = 2; i <= parts.length; i++) {
      const parent = parts.slice(0, i - 1).join('/');
      const child = parts.slice(0, i).join('/');
      dirMap[parent].childPaths.add(child);
    }
  }

  // Convert to tree nodes, stripping the common prefix
  const prefixParts = prefix ? prefix.split('/') : [];
  const prefixDepth = prefixParts.length;

  const nodeMap = {};
  for (const [path, info] of Object.entries(dirMap)) {
    const depth = path.split('/').length;
    if (depth <= prefixDepth) continue;

    // Count total tracks recursively
    const countDescendants = (p) => {
      const d = dirMap[p];
      if (!d) return 0;
      let c = d.trackCount;
      for (const cp of d.childPaths) c += countDescendants(cp);
      return c;
    };
    const totalTracks = countDescendants(path);

    const node = {
      id: 'lib:' + path,
      path,
      name: info.name,
      children: [],
      track_count: totalTracks,
      directTracks: info.trackCount,
    };
    nodeMap[path] = node;
  }

  // Build parent-child links
  for (const [path, info] of Object.entries(dirMap)) {
    if (!nodeMap[path]) continue;
    for (const cp of info.childPaths) {
      if (nodeMap[cp]) nodeMap[path].children.push(nodeMap[cp]);
    }
    nodeMap[path].children.sort((a, b) => a.name.localeCompare(b.name));
  }

  // Roots = nodes at prefixDepth + 1
  const roots = [];
  for (const [path, node] of Object.entries(nodeMap)) {
    const depth = path.split('/').length;
    if (depth === prefixDepth + 1) roots.push(node);
  }
  roots.sort((a, b) => a.name.localeCompare(b.name));

  return { roots, tracksByPath };
}
