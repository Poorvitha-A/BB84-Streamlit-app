"""
Advanced Quantum State Encoding Visualization for BB84 Protocol
Shows how bits are encoded into quantum states on the Bloch sphere.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_bloch_sphere():
    """Generate a Bloch sphere mesh for visualization."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def get_bloch_coordinates(bit, basis):
    """
    Get 3D Bloch sphere coordinates for a BB84 encoded state.
    
    BB84 Encoding:
    - Z-basis (rectilinear): |0⟩ = north pole, |1⟩ = south pole
    - X-basis (diagonal): |+⟩ = equator right, |-⟩ = equator left
    """
    if basis == 0:  # Z-basis (rectilinear)
        if bit == 0:
            return 0, 0, 1, "|0⟩"  # North pole
        else:
            return 0, 0, -1, "|1⟩"  # South pole
    else:  # X-basis (diagonal)
        if bit == 0:
            return 1, 0, 0, "|+⟩"  # Equator right
        else:
            return -1, 0, 0, "|-⟩"  # Equator left


def get_basis_label(basis):
    """Get human-readable basis label."""
    return "Z-basis (Rectilinear)" if basis == 0 else "X-basis (Diagonal)"


def get_basis_color(basis):
    """Get color for basis type."""
    return "rgba(30, 60, 200, 0.8)" if basis == 0 else "rgba(200, 100, 30, 0.8)"


def get_state_color(bit, basis):
    """Get color for quantum state (more saturated for visual distinction)."""
    if basis == 0:  # Z-basis
        return "rgba(0, 100, 255, 1)" if bit == 0 else "rgba(0, 0, 150, 1)"
    else:  # X-basis
        return "rgba(255, 100, 150, 1)" if bit == 0 else "rgba(200, 50, 50, 1)"


def create_quantum_state_encoding_visualization(alice_bits, alice_bases):
    """
    Create advanced Bloch sphere visualization of BB84 quantum state encoding.
    
    Parameters:
    -----------
    alice_bits : list
        Alice's random bits (0 or 1)
    alice_bases : list
        Alice's random bases (0=Z-basis, 1=X-basis)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D Bloch sphere showing encoded quantum states
    """
    
    # Create figure with multiple traces
    fig = go.Figure()
    
    # Generate Bloch sphere surface
    x, y, z = create_bloch_sphere()
    
    # Add semi-transparent Bloch sphere
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.15,
        colorscale=[[0, 'rgba(200, 200, 200, 0.1)'], [1, 'rgba(150, 150, 150, 0.1)']],
        showscale=False,
        hoverinfo='skip',
        name='Bloch Sphere',
    ))
    
    # Add coordinate axes
    axis_length = 1.3
    
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='rgba(255, 100, 100, 0.6)', width=3),
        name='X-axis (|+⟩/|-⟩)', hoverinfo='skip'
    ))
    
    # Y-axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='rgba(100, 255, 100, 0.6)', width=3),
        name='Y-axis', hoverinfo='skip'
    ))
    
    # Z-axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='rgba(100, 100, 255, 0.6)', width=3),
        name='Z-axis (|0⟩/|1⟩)', hoverinfo='skip'
    ))
    
    # Add basis labels at axis ends
    fig.add_trace(go.Scatter3d(
        x=[axis_length * 1.15], y=[0], z=[0],
        mode='text', text=['|+⟩/|-⟩'],
        textposition='top center',
        textfont=dict(size=12, color='rgba(255, 100, 100, 0.8)'),
        hoverinfo='skip', showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[axis_length * 1.15],
        mode='text', text=['|0⟩ (top)'],
        textposition='top center',
        textfont=dict(size=12, color='rgba(100, 100, 255, 0.8)'),
        hoverinfo='skip', showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-axis_length * 1.15],
        mode='text', text=['|1⟩ (bottom)'],
        textposition='top center',
        textfont=dict(size=12, color='rgba(100, 100, 255, 0.8)'),
        hoverinfo='skip', showlegend=False
    ))
    
    # Add encoded quantum states
    z_basis_states = {'x': [], 'y': [], 'z': [], 'color': [], 'text': [], 'bit': [], 'basis': []}
    x_basis_states = {'x': [], 'y': [], 'z': [], 'color': [], 'text': [], 'bit': [], 'basis': []}
    
    for i, (bit, basis) in enumerate(zip(alice_bits, alice_bases)):
        x_coord, y_coord, z_coord, state_label = get_bloch_coordinates(bit, basis)
        color = get_state_color(bit, basis)
        hover_text = f"<b>Bit #{i+1}</b><br>Value: {bit}<br>Basis: {get_basis_label(basis)}<br>State: {state_label}"
        
        if basis == 0:  # Z-basis
            z_basis_states['x'].append(x_coord)
            z_basis_states['y'].append(y_coord)
            z_basis_states['z'].append(z_coord)
            z_basis_states['color'].append(color)
            z_basis_states['text'].append(hover_text)
            z_basis_states['bit'].append(bit)
            z_basis_states['basis'].append(basis)
        else:  # X-basis
            x_basis_states['x'].append(x_coord)
            x_basis_states['y'].append(y_coord)
            x_basis_states['z'].append(z_coord)
            x_basis_states['color'].append(color)
            x_basis_states['text'].append(hover_text)
            x_basis_states['bit'].append(bit)
            x_basis_states['basis'].append(basis)
    
    # Add Z-basis states
    if z_basis_states['x']:
        fig.add_trace(go.Scatter3d(
            x=z_basis_states['x'], y=z_basis_states['y'], z=z_basis_states['z'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=z_basis_states['color'],
                line=dict(color='white', width=2),
                opacity=1
            ),
            text=[f"<b>{i+1}</b>" for i in range(len(z_basis_states['x']))],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            hovertext=z_basis_states['text'],
            hoverinfo='text',
            name='Z-basis States',
            showlegend=True
        ))
    
    # Add X-basis states
    if x_basis_states['x']:
        fig.add_trace(go.Scatter3d(
            x=x_basis_states['x'], y=x_basis_states['y'], z=x_basis_states['z'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=x_basis_states['color'],
                line=dict(color='white', width=2),
                opacity=1
            ),
            text=[f"<b>{i+1}</b>" for i in range(len(x_basis_states['x']))],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            hovertext=x_basis_states['text'],
            hoverinfo='text',
            name='X-basis States',
            showlegend=True
        ))
    
    # Update layout for professional appearance
    fig.update_layout(
        title={
            'text': '<b>BB84 Quantum State Encoding on Bloch Sphere</b><br><sub>Each bit is encoded as a quantum state in one of two bases</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis=dict(
                title='X (Diagonal Basis)',
                backgroundcolor='rgba(240, 240, 240, 0.5)',
                gridcolor='rgba(200, 200, 200, 0.3)',
                showbackground=True,
                range=[-1.5, 1.5]
            ),
            yaxis=dict(
                title='Y',
                backgroundcolor='rgba(240, 240, 240, 0.5)',
                gridcolor='rgba(200, 200, 200, 0.3)',
                showbackground=True,
                range=[-1.5, 1.5]
            ),
            zaxis=dict(
                title='Z (Rectilinear Basis)',
                backgroundcolor='rgba(240, 240, 240, 0.5)',
                gridcolor='rgba(200, 200, 200, 0.3)',
                showbackground=True,
                range=[-1.5, 1.5]
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(100, 100, 100, 0.3)',
            borderwidth=1
        ),
        font=dict(family='Arial, sans-serif', size=12),
        hovermode='closest',
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig


def create_encoding_explanation_table(alice_bits, alice_bases):
    """
    Create a detailed table showing bit encoding details.
    
    Parameters:
    -----------
    alice_bits : list
        Alice's random bits (0 or 1)
    alice_bases : list
        Alice's random bases (0=Z-basis, 1=X-basis)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Table showing encoding details
    """
    
    bit_numbers = [f"#{i+1}" for i in range(len(alice_bits))]
    bits = [str(b) for b in alice_bits]
    bases = [get_basis_label(b) for b in alice_bases]
    states = []
    
    for bit, basis in zip(alice_bits, alice_bases):
        _, _, _, state_label = get_bloch_coordinates(bit, basis)
        states.append(state_label)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Bit #</b>', '<b>Value</b>', '<b>Basis</b>', '<b>Quantum State</b>'],
            fill_color='rgba(0, 100, 200, 0.7)',
            align='center',
            font=dict(color='white', size=13, family='Arial, sans-serif')
        ),
        cells=dict(
            values=[bit_numbers, bits, bases, states],
            fill_color='rgba(240, 240, 250, 0.5)',
            align='center',
            font=dict(size=12, family='Arial, sans-serif'),
            height=25
        )
    )])
    
    fig.update_layout(
        title={
            'text': '<b>BB84 Encoding Details</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        width=600,
        height=min(400 + len(alice_bits) * 20, 800),
        margin=dict(l=20, r=20, t=70, b=20)
    )
    
    return fig
