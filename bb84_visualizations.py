# BB84 Visualizations Module - Advanced charting and visualization
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch
from qiskit.quantum_info import Statevector
import bb84_config as config

# ============================================================
# PDF STYLE TIMELINE PLOTS
# ============================================================
def plot_pdf_style_timeline(timeline_df, title="BB84 Timeline", max_bits=50):
    """Create professional PDF-style timeline visualization
    
    Args:
        timeline_df: Timeline DataFrame
        title: Plot title
        max_bits: Maximum bits to display
    
    Returns:
        matplotlib.figure.Figure: Timeline figure
    """
    display_df = timeline_df.head(max_bits)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9))

    # Plot 1: Alice's bits with visibility
    colors1 = []
    edge_colors1 = []
    heights1 = []

    for bit in display_df['AliceBit']:
        if bit == 0:
            colors1.append('lightblue')
            edge_colors1.append('blue')
            heights1.append(0.2)
        else:
            colors1.append('darkblue')
            edge_colors1.append('navy')
            heights1.append(1.0)

    ax1.bar(display_df['BitIndex'], heights1, color=colors1, 
            edgecolor=edge_colors1, linewidth=1, alpha=0.9)
    ax1.set_title(f'{title} - Alice\'s Transmitted Bits')
    ax1.set_ylabel('Bit Value')
    ax1.set_ylim(0, 1.2)

    # Add text labels
    for idx, bit, height in zip(display_df['BitIndex'], display_df['AliceBit'], heights1):
        label_y = height + 0.05 if height < 0.5 else height - 0.15
        text_color = 'black' if height < 0.5 else 'white'
        ax1.text(idx, label_y, str(int(bit)), ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_color)

    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Plot 2: Base matching
    colors2 = ['limegreen' if m else 'crimson' for m in display_df['BaseMatch']]
    heights2 = [1.0] * len(display_df)
    ax2.bar(display_df['BitIndex'], heights2, color=colors2, 
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_title('Base Matching (Green=Match, Red=Mismatch)')
    ax2.set_ylabel('Match Status')
    ax2.set_ylim(0, 1.2)

    # Plot 3: Error status
    error_heights = []
    colors3 = []

    for _, row in display_df.iterrows():
        if row['Used']:
            if row['Error']:
                error_heights.append(1.0)
                colors3.append('red')
            else:
                error_heights.append(0.8)
                colors3.append('forestgreen')
        else:
            error_heights.append(0.3)
            colors3.append('lightgray')

    ax3.bar(display_df['BitIndex'], error_heights, color=colors3,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax3.set_title('Transmission Results')
    ax3.set_xlabel('Bit Index')
    ax3.set_ylabel('Status')
    ax3.set_ylim(0, 1.2)

    legend_elements = [
        Patch(facecolor='forestgreen', alpha=0.8, label='Correct'),
        Patch(facecolor='red', alpha=0.8, label='Error'),
        Patch(facecolor='lightgray', alpha=0.8, label='Not Used')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig

# ============================================================
# PLOTLY INTERACTIVE TIMELINES
# ============================================================
def plotly_bit_timeline(timeline_df, start, end, title="Bit Timeline"):
    """Create interactive bit timeline
    
    Args:
        timeline_df: Timeline DataFrame
        start: Start index
        end: End index
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    sliced = timeline_df[(timeline_df["BitIndex"] >= start) & (timeline_df["BitIndex"] <= end)].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sliced["BitIndex"], y=sliced["AliceBit"],
        mode="lines+markers", name="Alice Bit",
        marker=dict(color='#1E40AF', size=6),
        hovertemplate="Index=%{x}<br>Alice=%{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=sliced["BitIndex"], y=sliced["BobResult"],
        mode="lines+markers", name="Bob Result",
        marker=dict(color='#9CA3AF', size=6),
        hovertemplate="Index=%{x}<br>Bob=%{y}<extra></extra>"
    ))

    err = sliced[(sliced["Used"] == True) & (sliced["Error"] == True)]
    if len(err) > 0:
        fig.add_trace(go.Scatter(
            x=err["BitIndex"], y=err["BobResult"],
            mode="markers", name="Errors",
            marker=dict(size=12, symbol="x", color='red'),
            hovertemplate="Index=%{x}<br>ERROR<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Bit Index",
        yaxis_title="Bit (0/1)",
        yaxis=dict(tickmode="array", tickvals=[0, 1]),
        height=config.TIMELINE_HEIGHT,
        hovermode='x unified'
    )
    return fig

def plotly_error_timeline(timeline_df, start, end, title="Error Timeline"):
    """Create interactive error timeline
    
    Args:
        timeline_df: Timeline DataFrame
        start: Start index
        end: End index
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Interactive error plot
    """
    sliced = timeline_df[(timeline_df["BitIndex"] >= start) & (timeline_df["BitIndex"] <= end)].copy()
    y = ((sliced["Used"] == True) & (sliced["Error"] == True)).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sliced["BitIndex"],
        y=y,
        name="Error",
        marker=dict(color='red')
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Bit Index",
        yaxis_title="Error Present",
        height=config.ERROR_TIMELINE_HEIGHT,
        showlegend=False
    )
    return fig

# ============================================================
# QBER AND SECURITY VISUALIZATIONS
# ============================================================
def qber_gauge(qber, threshold, label="QBER Attack Detection"):
    """Create QBER gauge chart
    
    Args:
        qber: Quantum bit error rate
        threshold: QBER threshold
        label: Label for the gauge (e.g., "Without Eve", "With Eve")
    
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=qber,
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [0, 0.25]},
            'threshold': {'line': {'width': 4}, 'value': threshold},
            'steps': [
                {'range': [0, threshold], 'color': "lightgreen"},
                {'range': [threshold, 0.25], 'color': "lightcoral"}
            ]
        },
        title={'text': label},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=config.GAUGE_HEIGHT)
    return fig

def decision_line(qber, threshold, title="Decision Graph"):
    """Create QBER decision line visualization
    
    Args:
        qber: Quantum bit error rate
        threshold: QBER threshold
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Decision chart
    """
    x = np.linspace(0, 0.25, 300)
    y = np.ones_like(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", 
                            name="Decision Line", line=dict(color='blue')))
    fig.add_vline(x=threshold, line_dash="dash", 
                 annotation_text="Threshold", annotation_position="top",
                 line_color='orange')
    fig.add_vline(x=qber, line_dash="solid", 
                 annotation_text="QBER", annotation_position="bottom",
                 line_color='red')
    
    fig.update_layout(title=title, xaxis_title="QBER", 
                     yaxis=dict(visible=False), height=240)
    return fig

def qber_attack_detection_comparison(qber_no_eve, qber_eve, threshold):
    """Create parallel QBER comparison for attack detection (With/Without Eve)
    
    Args:
        qber_no_eve: QBER without Eve
        qber_eve: QBER with Eve
        threshold: Detection threshold
    
    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    fig = go.Figure()
    
    # Add QBER bars
    scenarios = ['Without Eve', 'With Eve']
    qber_values = [qber_no_eve, qber_eve]
    colors = ['green' if q <= threshold else 'red' for q in qber_values]
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=qber_values,
        marker=dict(color=colors, line=dict(color='black', width=2)),
        text=[f'{q:.4f}' for q in qber_values],
        textposition='outside',
        name='QBER',
        showlegend=False
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                  annotation_text=f"Threshold ({threshold:.4f})",
                  annotation_position="right")
    
    fig.update_layout(
        title="🔒 QBER Attack Detection Comparison",
        xaxis_title="Scenario",
        yaxis_title="QBER Value",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, max(max(qber_values), threshold) * 1.2])
    )
    
    return fig


# ============================================================
# ADVANCED BLOCH SPHERE VISUALIZATION
# ============================================================
def plotly_bloch_sphere(states, title="Bloch Sphere"):
    """Create advanced 3D Bloch sphere visualization
    
    Args:
        states: List of Statevector or (theta, phi) tuples
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: 3D Bloch sphere
    """
    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    
    # Add sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.15, colorscale='Blues', showscale=False,
        name='Bloch Sphere'
    ))

    # Add coordinate axes with enhanced styling
    axis_configs = [
        ([-1.3, 1.3], [0, 0], [0, 0], 'red', 'X-axis'),
        ([0, 0], [-1.3, 1.3], [0, 0], 'green', 'Y-axis'),
        ([0, 0], [0, 0], [-1.3, 1.3], 'blue', 'Z-axis')
    ]
    
    for x, y, z, color, name in axis_configs:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=4),
            name=name,
            hoverinfo='skip'
        ))

    # Add axis labels
    labels = [
        (1.2, 0, 0, 'X', 'green'),
        (0, 1.2, 0, 'Y', 'blue'),
        (0, 0, 1.2, 'Z', 'red')
    ]
    
    for x, y, z, text, color in labels:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='text',
            text=[text],
            textfont=dict(size=14, color=color),
            hoverinfo='skip'
        ))

    # Add quantum states
    colors_palette = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime']
    
    for i, sv in enumerate(states):
        if hasattr(sv, 'data'):
            probs = np.abs(sv.data) ** 2
            if len(probs) == 2:
                theta = 2 * np.arccos(np.sqrt(probs[0]))
                phi = np.angle(sv.data[1]) - np.angle(sv.data[0])
            else:
                continue
        else:
            theta, phi = sv

        x_p = np.sin(theta) * np.cos(phi)
        y_p = np.sin(theta) * np.sin(phi)
        z_p = np.cos(theta)

        color = colors_palette[i % len(colors_palette)]
        
        # Add state point
        fig.add_trace(go.Scatter3d(
            x=[x_p], y=[y_p], z=[z_p],
            mode='markers+text+lines',
            marker=dict(size=10, color=color, symbol='diamond'),
            text=[f'Q{i}'],
            textfont=dict(size=12, color=color),
            textposition="top center",
            name=f'Qubit {i}',
            hovertemplate=f'Qubit {i}<br>θ={theta:.2f}<br>φ={phi:.2f}<extra></extra>',
            line=dict(color=color, width=2)
        ))
        
        # Add vector from origin to state
        fig.add_trace(go.Scatter3d(
            x=[0, x_p], y=[0, y_p], z=[0, z_p],
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            zaxis=dict(showgrid=True, zeroline=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=config.BLOCH_SPHERE_HEIGHT,
        showlegend=True,
        hovermode='closest'
    )
    return fig

# ============================================================
# ADVANCED METRICS VISUALIZATIONS
# ============================================================
def create_metrics_dashboard(no_eve_metrics, eve_metrics):
    """Create comprehensive metrics comparison dashboard
    
    Args:
        no_eve_metrics: Metrics for no Eve scenario
        eve_metrics: Metrics for with Eve scenario
    
    Returns:
        plotly.graph_objects.Figure: Comparison figure
    """
    fig = go.Figure()

    metrics_list = ['Sifted', 'Correct', 'Errors', 'QBER×100']
    
    no_eve_values = [
        no_eve_metrics['sifted_count'],
        no_eve_metrics['correct_count'],
        no_eve_metrics['error_count'],
        no_eve_metrics['qber'] * 100
    ]
    
    eve_values = [
        eve_metrics['sifted_count'],
        eve_metrics['correct_count'],
        eve_metrics['error_count'],
        eve_metrics['qber'] * 100
    ]

    fig.add_trace(go.Bar(
        name='No Eve',
        x=metrics_list,
        y=no_eve_values,
        marker_color='lightgreen'
    ))

    fig.add_trace(go.Bar(
        name='With Eve',
        x=metrics_list,
        y=eve_values,
        marker_color='salmon'
    ))

    fig.update_layout(
        title="Metrics Comparison: No Eve vs With Eve",
        barmode='group',
        hovermode='x unified',
        height=400
    )
    return fig

def create_efficiency_chart(num_bits, sift_no, sift_e, key_no, key_e):
    """Create efficiency and key rate visualization
    
    Args:
        num_bits: Total qubits transmitted
        sift_no: Sifted bits (no Eve)
        sift_e: Sifted bits (with Eve)
        key_no: Final key (no Eve)
        key_e: Final key (with Eve)
    
    Returns:
        plotly.graph_objects.Figure: Efficiency chart
    """
    stages = ['Transmitted', 'Sifted', 'Final Key']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stages,
        y=[num_bits, sift_no, key_no],
        name='No Eve',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=stages,
        y=[num_bits, sift_e, key_e],
        name='With Eve',
        marker_color='lightcoral'
    ))

    fig.update_layout(
        title="Key Generation Pipeline Efficiency",
        barmode='group',
        xaxis_title="Stage",
        yaxis_title="Bit Count",
        height=400,
        hovermode='x unified'
    )
    return fig

# ============================================================
# PDF REPORT GENERATION
# ============================================================
def create_pdf_report_with_graphs(
    project_info: tuple,
    summary: tuple,
    timeline_df_no_eve_csv: str,
    timeline_df_eve_csv: str,
    num_bits: int,
    sift_no: int, key_no: int, qber_no: float,
    sift_e: int, key_e: int, qber_e: float,
    threshold: float,
    pdf_max_bits: int = 50
) -> bytes:
    """
    Creates a multi-section PDF report WITH graphs:
    - Page 1: Summary + metrics + conclusion
    - Page 2: PDF-style timeline (No Eve)
    - Page 3: PDF-style timeline (With Eve)
    - Page 4: Comparison bar chart
    """

    from matplotlib.backends.backend_pdf import PdfPages

    # Convert CSVs back to DataFrames
    timeline_df_no_eve = pd.read_csv(io.StringIO(timeline_df_no_eve_csv))
    timeline_df_eve = pd.read_csv(io.StringIO(timeline_df_eve_csv))
    
    # Convert tuple back to dict
    project_info = dict(project_info)
    summary = dict(summary)

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:

        # -------------------- PAGE 1 (Summary) --------------------
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4
        plt.axis("off")

        lines = []
        lines.append("AQVH FINAL PROJECT REPORT")
        lines.append("=" * 60)
        lines.append("PROJECT DETAILS")
        lines.append("-" * 60)

        for k, v in project_info.items():
            lines.append(f"{k}: {v}")

        lines.append("")
        lines.append("RESULT SUMMARY")
        lines.append("-" * 60)
        for k, v in summary.items():
            lines.append(f"{k}: {v}")

        lines.append("")
        lines.append("CONCLUSION")
        lines.append("-" * 60)
        if qber_e > threshold:
            lines.append("✅ Eavesdropping Detected (With Eve case).")
            lines.append("Reason: QBER exceeds threshold. Key exchange should be aborted.")
        else:
            lines.append("✅ Channel Secure. QBER below threshold. Key exchange successful.")
        
        lines.append("")
        lines.append("COMPARISON: No Eve vs With Eve")
        lines.append("vsk")

        fig1.text(0.06, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # -------------------- PAGE 2 (Timeline No Eve) --------------------
        fig2 = plot_pdf_style_timeline(
            timeline_df_no_eve,
            title="BB84 Transmission Timeline - WITHOUT EVE",
            max_bits=pdf_max_bits
        )
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # -------------------- PAGE 3 (Timeline With Eve) --------------------
        fig3 = plot_pdf_style_timeline(
            timeline_df_eve,
            title="BB84 Transmission Timeline - WITH EVE",
            max_bits=pdf_max_bits
        )
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # -------------------- PAGE 4 (Comparison Bar Chart) --------------------
        fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 5))  # landscape feel

        ax1.bar(["Transmitted", "Sifted", "Final Key"], [num_bits, sift_no, key_no])
        ax1.set_title(f"No Eve (QBER: {qber_no:.4f})")
        ax1.set_ylabel("Number of Bits")

        ax2.bar(["Transmitted", "Sifted", "Final Key"], [num_bits, sift_e, key_e])
        ax2.set_title(f"With Eve (QBER: {qber_e:.4f})")
        ax2.set_ylabel("Number of Bits")

        plt.suptitle("Comparison: Without Eve vs With Eve", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        # Add "vsk" text at the bottom
        fig4.text(0.5, 0.02, "vsk", ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')
        
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    buf.seek(0)
    return buf.read()