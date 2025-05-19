import plotly.graph_objects as go
import numpy as np
from collections import deque
from piglet import Piglet
from pig import Pig


def plot_piglet_convergence(res: Piglet):
    """Convergence plot for Piglet

    Args:
        res (Piglet): Piglet class for getting trace of the value function

    Returns:
        fig: A plotly figure
    """
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

def plot_pig_policy(res: Pig):
    """Optimal policy visualisationi

    Args:
        res (Pig): Pig class containing the optimal policy

    Returns:
        fig: 3D isosurface representing the optimal policy
    """
    N = res.T + 1

    volume = np.zeros((N, N, N), dtype=np.uint8)

    # Optimal policy dictionary
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

def get_reachable_states(res: Pig, s_initial: tuple[int, int, int] = (0, 0, 0)):
    """Obtain reachable states given a initial condition

    Args:
        res (Pig): Pig class with the game information
        s_initial (tuple[int, int, int], optional): Initial. Defaults to (0, 0, 0).

    Returns:
        reachable_states: Reachable states with turn total equals to 0
    """
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

    return reachable_states

def plot_reachable_states(res: Pig, optimal=False):
    """Reachable state plot

    Args:
        res (Pig): Pig class with all the information
        optimal (bool, optional): Flag if consider the optimal policy. Defaults to False.

    Returns:
        fig: A plotly figure
    """
    # Reachable frontier cut by the optimal policy
    optimal_reachable_border = []
    for point, value in res.policy.items():
        if value == "roll":
            neighbour = (point[0], point[1], point[2] + 1)
            if res.policy.get(neighbour, "hold") == "hold":
                optimal_reachable_border.append(point)

    # Proper reachable frontier
    reachable_border = []
    for point in optimal_reachable_border:
        best_candidate = point
        for r in range(2, 7):  # r = 2 to 6
            candidate = (point[0], point[1], point[2] + r)
            if candidate[0]+candidate[2]< res.T - 1:  
                best_candidate = candidate  # keep updating with better (larger r)
        if best_candidate:
            reachable_border.append(best_candidate)

    # Considering the surface as a function
    max_points = {}
    if optimal:
        aux_border = optimal_reachable_border
    else:
        aux_border = reachable_border

    # Considering the surface as a function (only the minimum turn total)
    for x, y, z in aux_border:
        if (x, y) not in max_points or z <= max_points[(x, y)]:
            max_points[(x, y)] = z

    # Obtain reachable states, iterate over the opponent score
    reachable_states = []
    for j in range(res.T):
        aux = get_reachable_states(res, s_initial=(0, j, 0))
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

def plot_win_prob_contours(res:Pig, levels: tuple[float,...] =(0.03, 0.09, 0.27, 0.81)):
    """Plot contour plots

    Args:
        res (_ty): 
        levels (tuple, optional): _description_. Defaults to (0.03, 0.09, 0.27, 0.81).

    Returns:
        _type_: _description_
    """
    N = res.T
    # Build Vgrid with default -1 so only real values show
    Vgrid = np.full((N, N, N), -1.0, dtype=np.float32)
    # Vgrid =  np.zeros((N, N, N), dtype= np.float32)
    for (i, j, k), v in res.V.items():
        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
            Vgrid[i, j, k] = v

    x, y, z = np.mgrid[0:N, 0:N, 0:N]

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
            x=[0], y=[N-1], z=[round(lvl*N)],
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
        title="Win-probability contours for optimal play",
        scene=dict(
            xaxis_title="player 1 score",
            yaxis_title="player 2 score",
            zaxis_title="turn total",
            camera=camera,
            # xaxis=dict(nticks=5, range=[0,100]),
            # yaxis=dict(nticks=5, range=[0,100]),
            # zaxis=dict(nticks=5, range=[0,100]),
        ),
        template="plotly_white",
        margin=dict(l=0,r=0,b=0,t=30),
    )
    return fig

def plot_cross_section(res:Pig, section: int = 30):
    """Cross section of the reachable states and optimal policy border

    Args:
        res (Pig): Pig class containing the optimal policy
        section (int, optional): Opponent score section. Defaults to 30.
    """
    optimal_border_section = []
    for point, value in res.policy.items():
        if value == "roll" and point[1]==section:
            neighbour = (point[0], point[1], point[2] + 1)
            if res.policy.get(neighbour, "hold") == "hold":
                optimal_border_section.append(point)

    optimal_reachable_section = []
    for point in optimal_border_section:
        best_candidate = point
        for r in range(2, 7):  # r = 2 to 6
            candidate = (point[0], point[1], point[2] + r)
            if candidate[0]+candidate[2]< res.T-1:  # your condition here
                best_candidate = candidate  # keep updating with better (larger r)
        if best_candidate:
            optimal_reachable_section.append(best_candidate)

    # Consider optimal border 6 points above
    aux = get_reachable_states(res, s_initial=(0, section, 0))

    # Consider the part of the border as a function
    max_points = {}
    for x, y, z in optimal_reachable_section:
        if (x, y) not in max_points or z <= max_points[(x, y)]:
            max_points[(x, y)] = z

    reachable_states_section = np.array([
                (point[0], max_points[point[0],point[1]])
                for point in aux
            ])

    # Array for a optimal threshold section
    array_border = np.array([(point[0],point[2]) for point in optimal_border_section]) # lineplot
    array_border = array_border[array_border[:, 0].argsort()]
    if section == 30:
        min_val, max_val = 59, 74    
        rows_indices = (array_border[:, 0] >= min_val) & (array_border[:, 0] <= max_val)
        rows_in_range = array_border[rows_indices]
        aux_values = np.unique(rows_in_range[:,0])
        for v in aux_values:
            idx = rows_in_range[:,0]==v
            subset = rows_in_range[idx]
            subset = subset[subset[:,1].argsort()]
            rows_in_range[idx] = subset

        rows_in_range = np.concatenate([rows_in_range[::2],rows_in_range[1::2]])
        array_border[rows_indices] = rows_in_range

    # Array of reachable states given a section
    array_reachable_states = np.array(reachable_states_section) # barplot
    array_reachable_states = array_reachable_states[array_reachable_states[:, 0].argsort()]

    # Create figure
    fig = go.Figure()

    # Add bar trace
    fig.add_trace(go.Bar(
        x=array_reachable_states[:,0],
        y=array_reachable_states[:,1],
        name='Reachable states',
        marker_color='lightblue'
    ))

    # Add line trace
    fig.add_trace(go.Scatter(
        x=array_border[:,0],
        y=array_border[:,1],
        name='Threshold policy',
        mode='lines',
        line=dict(color='black', width=2),
        marker=dict(size=6)
    ))

    # Add horizontal dashed line at y = 50 (change this value as needed)
    fig.add_trace(go.Scatter(
        x=array_reachable_states[:,0],
        y=[20]*array_reachable_states.shape[0],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Hold 20'  # This will appear in the legend
    ))

    # Layout
    fig.update_layout(
        title=f'Cross section for the threshold policy with j={section}',
        yaxis_title='turn total',
        xaxis_title='player 1 score',
        barmode='group',  # optional: group bars if multiple
        template='plotly_white'
    )

    return fig

############
# Appendix #
############
def plot_expected_turns(avg_turns):
    """Plot expected turns until end

    Args:
        avg_turns (_type_): Vector with expected turns to win

    Returns:
        fig: A plotly figure
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
    """Average margin between players in victory

    Args:
        avg_margin (_type_): Vector with average margins

    Returns:
        fig: A plotly figure
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
