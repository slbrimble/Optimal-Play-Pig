import plotly.graph_objects as go
import numpy as np
from collections import deque
from piglet import Piglet
from pig import Pig


# Convergence plot Piglet
def plot_piglet_convergence(res: Piglet):
    curves_dict = res.trace

    x = list(range(len(next(iter(curves_dict.values())))))

    fig = go.Figure()

    # Add each curve to the figure
    for name, y_values in curves_dict.items():
        fig.add_trace(go.Scatter(x=x, y=y_values, mode="lines", name=str(name)))

    # Customise layout
    fig.update_layout(
        title=f"Convergence of Piglet with target T = {res.T}",
        xaxis_title="iteration",
        yaxis_title="probability",
        template="plotly_white",
    )
    return fig


# Optimal policy Pig plot
def plot_pig_policy(res: Pig):
    N = res.T + 1

    volume = np.zeros((N, N, N), dtype=np.uint8)

    # Fill your dictionary here
    decisions = res.policy

    for (x, y, z), val in decisions.items():
        if 0 <= x < N and 0 <= y < N and 0 <= z < N:
            volume[x, y, z] = 1 if val == "roll" else 0

    x, y, z = np.mgrid[0:N, 0:N, 0:N]

    # Plot
    fig = go.Figure()

    # Decision Boundary Surface (where roll meets hold)
    fig.add_trace(
        go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=0.5,
            isomax=0.5,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Reds",
            opacity=1,
            showscale=False,
            name="Decision Boundary",
        )
    )

    fig.update_layout(
        title=f"Optimal policy for Pig using value iteration with target T={res.T}",
        scene=dict(
            xaxis_title="player 1 score",
            yaxis_title="player 2 Score",
            zaxis_title="turn total",
        ),
        margin=dict(l=20, r=0, b=0, t=40),
    )

    return fig


# Reachable state plot
def plot_reachable_states(res: Pig, optimal=False):
    # Define auxiliar function for determining reachable states
    def get_reachable_states(res: Pig, s_initial: tuple[int, int, int] = (0, 0, 0)):
        reachable_states = list()
        visited = set()
        queue = deque([s_initial])

        while queue:
            s = queue.popleft()
            if s in visited:
                continue
            visited.add(s)
            reachable_states.append(s)

            t, p, u = s
            for k in range(res.T - t):
                if res.policy.get((t, p, k)) == "hold":
                    t_next = t + k
                    if t_next < res.T:
                        next_states = {(t_next + r, p, 0) for r in range(6)}
                        for ns in next_states:
                            if ns not in visited:
                                queue.append(ns)
                    break

        # Find border gaps
        reachable_scores = np.sort(np.array([s[0] for s in reachable_states]))
        breaks = np.where(np.diff(reachable_scores) > 1)[0]

        # Start and end indices of consecutive blocks
        start_indices = np.insert(breaks + 1, 0, 0)
        end_indices = np.append(breaks, len(reachable_scores) - 1)

        # Get start and end values of each block
        boundaries = []
        for start, end in zip(start_indices, end_indices):
            if start == end:
                boundaries.append(reachable_scores[start])
            else:
                boundaries.extend([reachable_scores[start], reachable_scores[end]])

        return reachable_states, boundaries

    # Reachable frontier cut by the optimal policy
    optimal_reachable_border = []
    for point, value in res.policy.items():
        if value == "roll":
            neighbour = (point[0], point[1], point[2] + 1)
            if res.policy.get(neighbour, "hold") == "hold":
                optimal_reachable_border.append(point)

    # Proper reachable frontier
    reachable_border = [
        (point[0], point[1], point[2] + 6) for point in optimal_reachable_border
    ]

    # Considering the surface as a function
    max_points = {}
    if optimal:
        aux_border = optimal_reachable_border
    else:
        aux_border = reachable_border

    # Considering the surface as a function (only the minimum turn total)
    for x, y, z in aux_border:
        if (x, y) not in max_points or z > max_points[(x, y)]:
            max_points[(x, y)] = z

    # Obtain reachable states, iterate over the opponent score
    reachable_states = []
    for j in range(res.T):
        aux, _ = get_reachable_states(res, s_initial=(0, j, 0))
        reachable_states_j = [
            (point[0], point[1], r)
            for point in aux
            for r in range(max_points[(point[0], point[1])])
        ]
        reachable_states = reachable_states + reachable_states_j

    # Plot for the desire surface
    points = np.array(reachable_states)
    N = res.T + 6
    grid = np.zeros((N, N, N), dtype=np.float32)

    for x, y, z in points:
        grid[x, y, z] = 1

    x, y, z = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1], 0 : grid.shape[2]]

    fig = go.Figure(
        data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=grid.flatten(),
            isomin=0.5,  # threshold for surface
            isomax=1.0,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=1,
            colorscale="Reds",
            showscale=False,
        )
    )

    fig.update_layout(
        title=f"Reachable states for a optimal Pig player with target T={res.T}",
        scene=dict(
            xaxis_title="player 1 score",
            yaxis_title="player 2 Score",
            zaxis_title="turn total",
        ),
        margin=dict(l=20, r=0, b=0, t=40),
    )

    return fig


def plot_win_prob_contours(pig, levels=(0.03, 0.09, 0.27, 0.81)):
    """
    Plot win-probability contour planes at given levels,
    with shading and labels for each percentage.
    """
    T = pig.T
    # Build Vgrid with default -1 so only real values show
    Vgrid = np.full((T, T, T), -1.0, dtype=float)
    for (i, j, k), v in pig.V.items():
        if 0 <= i < T and 0 <= j < T and 0 <= k < T:
            Vgrid[i, j, k] = v

    x, y, z = np.mgrid[0:T, 0:T, 0:T]
    fig = go.Figure()

    # Gray shades from light (low level) to dark (high level)
    grays = ["#eeeeee", "#cccccc", "#888888", "#444444"]

    for lvl, gray in zip(levels, grays):
        fig.add_trace(go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=Vgrid.flatten(),
            isomin=lvl, isomax=lvl,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            opacity=0.7,
            colorscale=[[0, gray],[1, gray]],
            name=f"{int(lvl*100)}%"
        ))
        # Add a 3D text label at (i=0, j=T-1, k=round(lvl*T))
        fig.add_trace(go.Scatter3d(
            x=[0], y=[T-1], z=[round(lvl*T)],
            mode="text",
            text=[f"{int(lvl*100)}%"],
            textposition="middle center",
            textfont=dict(color=gray, size=14),
            showlegend=False
        ))

    # Set a fixed camera to match the paper’s perspective
    camera = dict(
        eye=dict(x=1.5, y=-1.8, z=0.8)
    )

    fig.update_layout(
        title="Figure 7: Win-Probability Contours for Optimal Play",
        scene=dict(
            xaxis_title="Player 1 score (i)",
            yaxis_title="Player 2 score (j)",
            zaxis_title="Turn total (k)",
            camera=camera,
            xaxis=dict(nticks=5, range=[0,100]),
            yaxis=dict(nticks=5, range=[0,100]),
            zaxis=dict(nticks=5, range=[0,100]),
        ),
        template="plotly_white",
        margin=dict(l=0,r=0,b=0,t=30),
    )
    return fig

def plot_expected_turns(avg_turns):
    """
    Figure 6: Average number of turns to win vs. starting score i.
    Uses Plotly for a consistent look with other figures.
    """
    x = np.arange(len(avg_turns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=avg_turns,
        mode='lines+markers',
        name='Avg turns',
        marker=dict(size=6),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="Figure 6: Average Turns to Victory",
        xaxis_title="Starting Score i",
        yaxis_title="Average Turns to Win",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

def plot_expected_margin(avg_margin):
    """
    Figure 7: Average margin of victory vs. starting score i.
    Uses Plotly for consistency.
    """
    x = np.arange(len(avg_margin))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=avg_margin,
        mode='lines+markers',
        name='Avg margin',
        marker=dict(size=6),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="Figure 7: Average Margin of Victory",
        xaxis_title="Starting Score i",
        yaxis_title="Average Margin",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
