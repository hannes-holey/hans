"""Phase 0: Stencil verification for TaylorHoodP2P1.

For each of the four P2 node types (even-even, odd-odd, even-odd, odd-even) we
independently derive the set of fine-grid offsets that can contribute to the
residual at an origin node, and compare against the hardcoded stencil in
`elements.py`.

Methodology
-----------
The residual at origin O is assembled by integrating over all triangles that
contain O as one of their 6 nodes.  Only nodes in those same triangles can
contribute to R(O) as trial function nodes.  So:

    stencil(O) = union over all triangles T that contain O of nodes(T)

A triangle is identified by (square SW corner S, triangle index t ∈ {0,1}).
nodes(S,t) = { S + square_node_offsets[idx_to_std[t,k]] for k in 0..5 }.

The key point: node types differ in how many triangles they belong to:
  - even-even (corner):     corner of 4 squares → 6 triangles
  - even-odd / odd-even (edge midpoint): shared by 2 triangles across 2 squares
  - odd-odd (centre):       centre of 1 square → 2 triangles (both in same square)

Grid convention: (col, row), i.e. (x, y) with x = column index.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from GaPFlow.fem_2d.elements import TaylorHoodP2P1


# ---------------------------------------------------------------------------
# Reference geometry from elements.py
# ---------------------------------------------------------------------------

SQUARE_NODE_OFFSETS = np.array([
    [0, 0], [2, 0], [0, 2], [2, 2],   # corners: SW, SE, NW, NE
    [0, 1], [1, 0], [1, 1],            # edge mid-points: W, S, centre
    [1, 2], [2, 1],                    # edge mid-points: N, E
])

IDX_TO_STD = np.array([
    [0, 1, 2, 4, 5, 6],   # triangle 0 (lower-left)
    [3, 2, 1, 8, 7, 6],   # triangle 1 (upper-right)
])


def _triangle_nodes(sq_x: int, sq_y: int, tri: int) -> list:
    """Return the 6 fine-grid (col, row) coords of triangle tri in square (sq_x, sq_y)."""
    return [(sq_x + SQUARE_NODE_OFFSETS[IDX_TO_STD[tri, k], 0],
             sq_y + SQUARE_NODE_OFFSETS[IDX_TO_STD[tri, k], 1])
            for k in range(6)]


def derive_stencil(origin_col: int, origin_row: int,
                   grid_radius: int = 4) -> set:
    """Derive the complete set of (dcol, drow) offsets that interact with origin.

    Searches all even-even square SW corners within ±grid_radius of the origin.
    SW corners must be even in both col and row (density grid positions).
    Returns offsets relative to origin (as used in stencil_* lists).
    """
    neighbours = set()

    # Start from the nearest even value below origin - grid_radius
    start_x = (origin_col - 2 * grid_radius) & ~1  # round down to even
    start_y = (origin_row - 2 * grid_radius) & ~1

    for sq_x in range(start_x, origin_col + 2 * grid_radius + 1, 2):
        for sq_y in range(start_y, origin_row + 2 * grid_radius + 1, 2):
            for tri in range(2):
                nodes = _triangle_nodes(sq_x, sq_y, tri)
                if (origin_col, origin_row) in nodes:
                    for n in nodes:
                        neighbours.add((n[0] - origin_col,
                                        n[1] - origin_row))
    return neighbours


def _origin_for_type(node_type: str):
    """Return a representative interior origin (col, row) for each node type.

    All four types appear within the square with SW corner at (4,4):
      - even-even: corner     (4,4)  — offset (0,0) in square
      - odd-odd:   centre     (5,5)  — offset (1,1) in square
      - even-odd:  W-edge     (4,5)  — offset (0,1) in square
      - odd-even:  S-edge     (5,4)  — offset (1,0) in square

    These are interior nodes: all neighbouring squares fit within the
    search radius, so no boundary effects.
    """
    if node_type == 'even-even':
        return (4, 4)
    elif node_type == 'odd-odd':
        return (5, 5)   # centre of square (4,4)
    elif node_type == 'even-odd':
        return (4, 5)   # W-edge of square (4,4); also E-edge of square (2,4)
    elif node_type == 'odd-even':
        return (5, 4)   # S-edge of square (4,4); also N-edge of square (4,2)
    else:
        raise ValueError(node_type)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStencilEvenEven:
    def test_stencil_matches_derived(self):
        origin = _origin_for_type('even-even')
        expected = derive_stencil(*origin)
        actual = set(map(tuple, TaylorHoodP2P1.stencil_even_even))
        _check(actual, expected, 'even-even', origin)

    def test_stencil_contains_origin(self):
        assert (0, 0) in set(map(tuple, TaylorHoodP2P1.stencil_even_even))


class TestStencilOddOdd:
    def test_stencil_matches_derived(self):
        origin = _origin_for_type('odd-odd')
        expected = derive_stencil(*origin)
        actual = set(map(tuple, TaylorHoodP2P1.stencil_odd_odd))
        _check(actual, expected, 'odd-odd', origin)

    def test_stencil_contains_origin(self):
        assert (0, 0) in set(map(tuple, TaylorHoodP2P1.stencil_odd_odd))


class TestStencilEvenOdd:
    def test_stencil_matches_derived(self):
        origin = _origin_for_type('even-odd')
        expected = derive_stencil(*origin)
        actual = set(map(tuple, TaylorHoodP2P1.stencil_even_odd))
        _check(actual, expected, 'even-odd', origin)

    def test_stencil_contains_origin(self):
        assert (0, 0) in set(map(tuple, TaylorHoodP2P1.stencil_even_odd))


class TestStencilOddEven:
    def test_stencil_matches_derived(self):
        origin = _origin_for_type('odd-even')
        expected = derive_stencil(*origin)
        actual = set(map(tuple, TaylorHoodP2P1.stencil_odd_even))
        _check(actual, expected, 'odd-even', origin)

    def test_stencil_contains_origin(self):
        assert (0, 0) in set(map(tuple, TaylorHoodP2P1.stencil_odd_even))


# ---------------------------------------------------------------------------
# Symmetry sanity checks
# ---------------------------------------------------------------------------

class TestStencilSymmetry:
    """Cross-stencil consistency checks."""

    def test_odd_odd_is_closed_under_negation(self):
        """odd-odd stencil should be symmetric: if (dx,dy) is in it, so is (-dx,-dy)."""
        s = set(map(tuple, TaylorHoodP2P1.stencil_odd_odd))
        for dx, dy in list(s):
            assert (-dx, -dy) in s, f"({dx},{dy}) in odd-odd but ({-dx},{-dy}) is not"

    def test_even_even_is_closed_under_negation(self):
        s = set(map(tuple, TaylorHoodP2P1.stencil_even_even))
        for dx, dy in list(s):
            assert (-dx, -dy) in s, f"({dx},{dy}) in even-even but ({-dx},{-dy}) is not"

    def test_no_duplicate_entries(self):
        for name, stencil in [
            ('even-even', TaylorHoodP2P1.stencil_even_even),
            ('odd-odd',   TaylorHoodP2P1.stencil_odd_odd),
            ('even-odd',  TaylorHoodP2P1.stencil_even_odd),
            ('odd-even',  TaylorHoodP2P1.stencil_odd_even),
        ]:
            tuples = [tuple(e) for e in stencil]
            assert len(tuples) == len(set(tuples)), f"Duplicate entry in {name} stencil"


# ---------------------------------------------------------------------------
# Debug plot (always produced, saved to file)
# ---------------------------------------------------------------------------

def _check(actual: set, expected: set, label: str, origin: tuple):
    """Assert actual == expected.  If they differ, save a diagnostic plot and fail."""
    if actual != expected:
        _plot_stencil_debug(actual, expected, label, origin)
    assert actual == expected, (
        f"Stencil mismatch for {label}:\n"
        f"  Missing from hardcoded: {sorted(expected - actual)}\n"
        f"  Extra in hardcoded:     {sorted(actual - expected)}\n"
        f"  (debug plot saved to /tmp/stencil_debug_{label.replace('-','_')}.png)"
    )


def _plot_stencil_debug(actual: set, expected: set, label: str, origin: tuple):
    """Save a side-by-side plot of the hardcoded vs derived stencils."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, data, title in [
        (axes[0], actual,   f"Hardcoded  ({label})"),
        (axes[1], expected, f"Derived    ({label})"),
    ]:
        _draw_stencil(ax, data, actual & expected, actual - expected,
                      expected - actual, origin, title)
    plt.tight_layout()
    path = f"/tmp/stencil_debug_{label.replace('-', '_')}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)


def _draw_stencil(ax, data: set, common: set, only_hardcoded: set,
                  only_derived: set, origin: tuple, title: str):
    """Draw one stencil panel."""
    ax.set_facecolor('white')

    # draw background grid squares
    for sq_x in range(-4, 5, 2):
        for sq_y in range(-4, 5, 2):
            rect = mpatches.Rectangle(
                (sq_x - 0.05, sq_y - 0.05), 2.1, 2.1,
                linewidth=1.0, edgecolor='#cccccc', facecolor='#f8f8f8')
            ax.add_patch(rect)

    # draw fine-grid dots for all nodes in the visible area
    for gx in range(-4, 5):
        for gy in range(-4, 5):
            ax.plot(gx, gy, 'o', color='#dddddd', markersize=4, zorder=1)

    # colour scheme
    colour_map = {}
    for p in common:
        colour_map[p] = '#2166ac'       # blue  — in stencil (correct)
    for p in only_hardcoded:
        colour_map[p] = '#d73027'       # red   — hardcoded but shouldn't be
    for p in only_derived:
        colour_map[p] = '#1a9850'       # green — missing from hardcoded

    all_marked = data | only_derived
    for p in all_marked:
        col = colour_map.get(p, '#2166ac')
        ax.scatter(p[0], p[1], c=col, s=300, zorder=3,
                   edgecolors='white', linewidths=0.8)
        if p != (0, 0):
            ax.annotate(f"({p[0]},{p[1]})", xy=(p[0], p[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7.5, color='#333333')

    # origin marker
    ax.scatter([0], [0], c='black', s=500, marker='*', zorder=5,
               edgecolors='white', linewidths=0.8)
    ax.annotate("origin", xy=(0, 0), xytext=(5, -12),
                textcoords='offset points', fontsize=7.5,
                color='black', fontweight='bold')

    legend = [
        mpatches.Patch(color='#2166ac', label='in stencil'),
        mpatches.Patch(color='#d73027', label='hardcoded only'),
        mpatches.Patch(color='#1a9850', label='missing'),
    ]
    ax.legend(handles=legend, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor='#aaaaaa')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(-4, 5))
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Δcol', fontsize=9)
    ax.set_ylabel('Δrow', fontsize=9)
    ax.axhline(0, color='#888888', linewidth=0.6, zorder=0)
    ax.axvline(0, color='#888888', linewidth=0.6, zorder=0)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Standalone plot of all four stencils (always generated when module is run)
# ---------------------------------------------------------------------------

def plot_all_stencils(save_path: str = "/tmp/stencils_all.png"):
    """Produce a 2×2 grid showing all four hardcoded stencils.

    Saved to `save_path`.  Call this directly or run the module as a script.
    """
    entries = [
        ('even-even', TaylorHoodP2P1.stencil_even_even),
        ('odd-odd',   TaylorHoodP2P1.stencil_odd_odd),
        ('even-odd',  TaylorHoodP2P1.stencil_even_odd),
        ('odd-even',  TaylorHoodP2P1.stencil_odd_even),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                             facecolor='white')
    axes = axes.flatten()
    for ax, (label, stencil) in zip(axes, entries):
        origin = _origin_for_type(label)
        derived = derive_stencil(*origin)
        actual  = set(map(tuple, stencil))
        common         = actual & derived
        only_hardcoded = actual - derived
        only_derived   = derived - actual
        _draw_stencil(ax, actual | derived, common,
                      only_hardcoded, only_derived, origin, label)

    fig.suptitle(
        "P2 stencils  (blue = in stencil, red = hardcoded-only, green = missing)",
        fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return save_path


if __name__ == '__main__':
    path = plot_all_stencils()
    print(f"Plot saved to {path}")
