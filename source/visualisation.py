import plotly.graph_objects as go
import numpy as np
from piglet import Piglet
from pig import Pig

# Convergence plot Piglet
def plot_piglet_convergence(res: Piglet):
    curves_dict = res.trace

    x = list(range(len(next(iter(curves_dict.values())))))

    fig = go.Figure()

    # Add each curve to the figure
    for name, y_values in curves_dict.items():
        fig.add_trace(go.Scatter(x=x, y=y_values, mode='lines', name=str(name)))

    # Customise layout
    fig.update_layout(
        title=f'Convergence of Piglet with target T = {res.T}',
        xaxis_title='iteration',
        yaxis_title='probability',
        template='plotly_white'
    )
    return fig

def plot_pig_policy(res: Pig):
    N = res.T+1 
    volume = np.zeros((N, N, N), dtype=np.uint8)

    # Fill your dictionary here
    decisions = res.policy

    # Populate the volume
    for (x, y, z), val in decisions.items():
        if 0 <= x < N and 0 <= y < N and 0 <= z < N:
            volume[x, y, z] = 1 if val == 'roll' else 0

    # Create coordinate axes
    x, y, z = np.mgrid[0:N, 0:N, 0:N]

    # Plot the isosurface where value = 0.5 (the decision boundary)
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin=0.5,
        isomax=0.5,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.5,
        showscale=False  
    ))

    fig.update_layout(
        title=f'Optimal policy for Pig using value iteration with target T={res.T}',
        scene=dict(
            xaxis_title='player 1 score',
            yaxis_title='player 2 Score',
            zaxis_title='turn total',
        ),
        margin=dict(l=20, r=0, b=0, t=40)
    )

    return fig
