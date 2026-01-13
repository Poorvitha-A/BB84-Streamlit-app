import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import time
import io
from datetime import datetime
import plotly.graph_objects as go

# Qiskit only for Bloch sphere
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit_aer import AerSimulator
from qiskit import transpile
from matplotlib.patches import Patch


# ============================================================
# ADVANCED BB84 SIMULATOR WITH QISKIT
# ============================================================
class BB84Simulator:
    def __init__(self):
        self.simulator = AerSimulator()

    @staticmethod
    def encode_qubit(bit, basis):
        qc = QuantumCircuit(1, 1)
        if int(bit) == 1:
            qc.x(0)
        if int(basis) == 1:
            qc.h(0)
        return qc

    def simulate_transmission(self, alice_bits, alice_bases, bob_bases, eve_present=False, eve_intercept_prob=0.5, noise_prob=0.0):
        alice_bits = [int(b) for b in list(alice_bits)]
        alice_bases = [int(b) for b in list(alice_bases)]
        bob_bases = [int(b) for b in list(bob_bases)]

        bob_results = []
        eve_results = [] if eve_present else None

        for bit, a_basis, b_basis in zip(alice_bits, alice_bases, bob_bases):
            qc = self.encode_qubit(bit, a_basis)

            # Eve intercept-resend (optional)
            if eve_present and float(np.random.random()) < float(eve_intercept_prob):
                eve_basis = int(np.random.randint(0, 2))
                if eve_basis == 1:
                    qc.h(0)
                qc.measure(0, 0)
                job_eve = self.simulator.run(transpile(qc, self.simulator), shots=1)
                res_eve = job_eve.result().get_counts()
                eve_bit = int(list(res_eve.keys())[0])
                eve_results.append(eve_bit)

                qc = QuantumCircuit(1, 1)
                if eve_bit == 1:
                    qc.x(0)
                if eve_basis == 1:
                    qc.h(0)

            if b_basis == 1:
                qc.h(0)
            qc.measure(0, 0)
            job = self.simulator.run(transpile(qc, self.simulator), shots=1)
            counts = job.result().get_counts()
            measured_bit = int(list(counts.keys())[0])

            # Add channel noise
            if noise_prob > 0 and np.random.random() < noise_prob:
                measured_bit = 1 - measured_bit  # flip bit

            bob_results.append(measured_bit)

        return bob_results, eve_results

    @staticmethod
    def privacy_amplification(sifted_key, error_rate, target_security_level=1e-6):
        sifted_key = [int(b) for b in list(sifted_key)]
        n = len(sifted_key)
        if n == 0:
            return []

        e = float(error_rate)
        if e <= 0.0 or e >= 1.0:
            h_eve = 0.0 if e in (0.0, 1.0) else 0.0
        else:
            h_eve = -e * np.log2(e) - (1 - e) * np.log2(1 - e)

        secure_length = n * (1 - h_eve) - 2 * np.log2(1 / float(target_security_level))
        secure_length = max(0, int(secure_length))

        if secure_length == 0:
            return []

        key_str = ''.join('1' if b == 1 else '0' for b in sifted_key)
        digest = hashlib.sha256(key_str.encode()).hexdigest()
        binary_hash = bin(int(digest, 16))[2:].zfill(256)
        final_bits = [int(b) for b in binary_hash[:secure_length]]

        return final_bits

    @staticmethod
    def assess_security(qber, threshold=0.11):
        if qber <= threshold:
            return {
                'status': 'SECURE',
                'message': f'QBER ({qber:.3f}) below threshold. Key exchange successful.',
                'action': 'PROCEED_WITH_KEY'
            }

        return {
            'status': 'INSECURE',
            'message': f'QBER ({qber:.3f}) exceeds threshold ({threshold}). Eavesdropping suspected.',
            'action': 'ABORT_AND_RETRY'
        }


# ============================================================
# TIMELINE DATAFRAME (as in PDF logic)
# ============================================================
def create_transmission_timeline(alice_bits, alice_bases, bob_bases, bob_results, title="BB84 Transmission Timeline"):
    """Create detailed timeline visualization of BB84 transmission"""

    # Calculate matching bases and errors
    matches = (alice_bases == bob_bases)
    idx = np.where(matches)[0]
    sifted_alice = alice_bits[idx]
    sifted_bob = np.array(bob_results)[idx]

    # Create timeline DataFrame
    timeline_data = []
    sifted_idx = 0

    for i in range(len(alice_bits)):
        row = {
            'BitIndex': i,
            'AliceBit': alice_bits[i],
            'AliceBasis': alice_bases[i],
            'BobBasis': bob_bases[i],
            'BobResult': bob_results[i],
            'BaseMatch': matches[i],
            'Used': matches[i],
            'Error': False,
            'ErrorType': 'None'
        }

        if matches[i]:  # Only check errors for matching bases
            if sifted_idx < len(sifted_alice):
                row['Error'] = (sifted_alice[sifted_idx] != sifted_bob[sifted_idx])
                if row['Error']:
                    row['ErrorType'] = 'Transmission Error'
                sifted_idx += 1
        else:
            row['ErrorType'] = 'Basis Mismatch'

        timeline_data.append(row)

    return pd.DataFrame(timeline_data)


# ============================================================
# PDF STYLE MATPLOTLIB TIMELINE (exactly like your PDF)
# ============================================================
def plot_pdf_style_timeline(timeline_df, title="BB84 Transmission Timeline (PDF Style)", max_bits=50):
    """Plot timeline showing transmitted bits and errors - FIXED COLOR VERSION WITH VISIBLE 0-BITS"""

    # Limit display for readability
    display_df = timeline_df.head(max_bits)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9))

    # Plot 1: Alice's transmitted bits - FIXED WITH VISIBLE 0-BITS
    colors1 = []
    edge_colors1 = []
    heights1 = []  # NEW: Custom heights for visibility

    for bit in display_df['AliceBit']:
        if bit == 0:
            colors1.append('lightblue')  # Light blue for 0-bits
            edge_colors1.append('blue')
            heights1.append(0.2)  # Small height to make 0-bits visible
        else:
            colors1.append('darkblue')  # Dark blue for 1-bits
            edge_colors1.append('navy')
            heights1.append(1.0)  # Full height for 1-bits

    ax1.bar(display_df['BitIndex'], heights1,
            color=colors1, edgecolor=edge_colors1, linewidth=1, alpha=0.9)
    ax1.set_title(f'{title} - Alice\'s Transmitted Bits (Light Blue=0, Dark Blue=1)')
    ax1.set_ylabel('Bit Representation')
    ax1.set_ylim(0, 1.2)

    # Add text labels to show actual bit values clearly
    for i, (idx, bit, height) in enumerate(zip(display_df['BitIndex'], display_df['AliceBit'], heights1)):
        label_y = height + 0.05 if height < 0.5 else height - 0.15  # Position labels appropriately
        text_color = 'black' if height < 0.5 else 'white'  # Contrast for visibility
        ax1.text(idx, label_y, str(int(bit)), ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_color)

    # Add horizontal reference lines for clarity
    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3, label='0-bit level')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='1-bit level')

    # Plot 2: Base matching - ENHANCED VISIBILITY
    colors2 = []
    for match in display_df['BaseMatch']:
        colors2.append('limegreen' if match else 'crimson')  # More vivid colors

    heights2 = [1.0] * len(display_df)
    ax2.bar(display_df['BitIndex'], heights2, color=colors2,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_title('Base Matching (Lime Green=Match/Used, Crimson=Mismatch/Discarded)')
    ax2.set_ylabel('Base Match Indicator')
    ax2.set_ylim(0, 1.2)

    # Plot 3: Errors in sifted bits - ENHANCED WITH BETTER COLORS
    error_heights = []
    colors3 = []

    for _, row in display_df.iterrows():
        if row['Used']:
            if row['Error']:
                error_heights.append(1.0)  # Show error as height 1
                colors3.append('red')
            else:
                error_heights.append(0.8)  # Show correct as height 0.8
                colors3.append('forestgreen')  # Changed from 'green' to 'forestgreen'
        else:
            error_heights.append(0.3)  # Show unused as height 0.3
            colors3.append('lightgray')  # Changed from 'gray' to 'lightgray'

    ax3.bar(display_df['BitIndex'], error_heights, color=colors3,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax3.set_title('Transmission Results (Red=Error, Forest Green=Correct, Light Gray=Not Used)')
    ax3.set_xlabel('Bit Index')
    ax3.set_ylabel('Status')
    ax3.set_ylim(0, 1.2)

    # Enhanced legend with better colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='forestgreen', alpha=0.8, label='Correct'),
        Patch(facecolor='red', alpha=0.8, label='Error'),
        Patch(facecolor='lightgray', alpha=0.8, label='Not Used')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    # Add legend for first plot showing bit value representation
    bit_legend_elements = [
        Patch(facecolor='cyan', alpha=0.9, edgecolor='blue', label='0-bit (height=0.2)'),
        Patch(facecolor='navy', alpha=0.9, edgecolor='darkblue', label='1-bit (height=1.0)')
    ]
    ax1.legend(handles=bit_legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


# PLOTLY TIMELINES (interactive)
def plotly_bit_timeline(timeline_df, start, end, title="Plotly Timeline (Alice vs Bob)"):
    sliced = timeline_df[(timeline_df["BitIndex"] >= start) & (timeline_df["BitIndex"] <= end)].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sliced["BitIndex"], y=sliced["AliceBit"],
        mode="lines+markers", name="Alice Bit",
        hovertemplate="Index=%{x}<br>Alice=%{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=sliced["BitIndex"], y=sliced["BobResult"],
        mode="lines+markers", name="Bob Result",
        hovertemplate="Index=%{x}<br>Bob=%{y}<extra></extra>"
    ))

    err = sliced[(sliced["Used"] == True) & (sliced["Error"] == True)]
    if len(err) > 0:
        fig.add_trace(go.Scatter(
            x=err["BitIndex"], y=err["BobResult"],
            mode="markers", name="Errors",
            marker=dict(size=12, symbol="x"),
            hovertemplate="Index=%{x}<br>ERROR<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Bit Index",
        yaxis_title="Bit (0/1)",
        yaxis=dict(tickmode="array", tickvals=[0, 1]),
        height=420
    )
    return fig


def plotly_error_timeline(timeline_df, start, end, title="Plotly Error Timeline (1=Error)"):
    sliced = timeline_df[(timeline_df["BitIndex"] >= start) & (timeline_df["BitIndex"] <= end)].copy()
    y = ((sliced["Used"] == True) & (sliced["Error"] == True)).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sliced["BitIndex"],
        y=y,
        name="Error"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Bit Index",
        yaxis_title="Error",
        height=280
    )
    return fig


# ============================================================
# QBER PLOTS
# ============================================================
def qber_gauge(qber, threshold):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=qber,
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [0, 0.25]},
            'threshold': {'line': {'width': 4}, 'value': threshold},
            'steps': [
                {'range': [0, threshold], 'color': "lightblue"},
                {'range': [threshold, 0.25], 'color': "lightcoral"}
            ]
        },
        title={'text': "QBER Attack Detection"}
    ))
    fig.update_layout(height=320)
    return fig


def decision_line(qber, threshold, title="Threshold Decision Graph"):
    x = np.linspace(0, 0.25, 300)
    y = np.ones_like(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Decision Line"))
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="Threshold", annotation_position="top")
    fig.add_vline(x=qber, line_dash="solid", annotation_text="QBER", annotation_position="bottom")
    fig.update_layout(title=title, xaxis_title="QBER", yaxis=dict(visible=False), height=240)
    return fig


def plotly_bloch_sphere(states):
    """
    Plot multiple states on a 3D Bloch sphere using Plotly.
    states: list of (theta, phi) or Statevector
    """
    import numpy as np

    # Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Blues', showscale=False))

    # Axes
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-1,1], mode='lines', line=dict(color='red', width=3), name='Z-axis'))
    fig.add_trace(go.Scatter3d(x=[-1,1], y=[0,0], z=[0,0], mode='lines', line=dict(color='green', width=3), name='X-axis'))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[-1,1], z=[0,0], mode='lines', line=dict(color='blue', width=3), name='Y-axis'))

    # Axis labels
    fig.add_trace(go.Scatter3d(x=[1.1], y=[0], z=[0], mode='text', text=['X'], textfont=dict(size=14, color='green'), name='X-label'))
    fig.add_trace(go.Scatter3d(x=[0], y=[1.1], z=[0], mode='text', text=['Y'], textfont=dict(size=14, color='blue'), name='Y-label'))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1.1], mode='text', text=['Z'], textfont=dict(size=14, color='red'), name='Z-label'))

    # States
    for i, sv in enumerate(states):
        if hasattr(sv, 'data'):
            # Statevector
            probs = np.abs(sv.data)**2
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

        fig.add_trace(go.Scatter3d(x=[x_p], y=[y_p], z=[z_p], mode='markers+text', marker=dict(size=10, color='red'), text=[f'Qubit {i}'], textfont=dict(size=10), name=f'Qubit {i}'))

    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), height=600)
    return fig


def get_statevector_from_bit_basis(bit, basis):
    qc = QuantumCircuit(1)
    if int(bit) == 1:
        qc.x(0)
    if int(basis) == 1:
        qc.h(0)
    return Statevector.from_instruction(qc)


def state_label(bit, basis):
    if int(basis) == 0:
        return "|0⟩ (Z)" if int(bit) == 0 else "|1⟩ (Z)"
    return "|+⟩ (X)" if int(bit) == 0 else "|−⟩ (X)"


# ============================================================
# PDF REPORT (using matplotlib -> pdf bytes)
# ============================================================
def create_pdf_report_with_graphs(
    project_info: dict,
    summary: dict,
    timeline_df_no_eve: pd.DataFrame,
    timeline_df_eve: pd.DataFrame,
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
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    buf.seek(0)
    return buf.read()



# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }
    .run-button {
        background: linear-gradient(135deg, #0f62fe 0%, #0e4f9e 100%) !important;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(76, 175, 80, 0.8); }
        100% { box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4); }
    }
    .control-button {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%) !important;
    }
    .danger-button {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%) !important;
    }
    .success-button {
        background: linear-gradient(135deg, #0f62fe 0%, #0a3a5c 100%) !important;
    }
    .warning-button {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: 600;
        border: 1px solid #e0e0e0;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        h2, h4 {
            text-align: center !important;
        }
        .stImage img {
            display: block;
            margin: 0 auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header (ONE LOGO)
    left, center = st.columns([4, 8])
    with left:
        try:
            st.image("jntua_logo.png", width=400)
        except Exception:
            st.write("Logo not found")

    with center:
        st.markdown(
            """
            <h2 style="text-align:left; margin-bottom:0;">
                Jawaharlal Nehru Technological University Anantapur College of Engineering Anantapur (JNTUACEA)
            </h2>
            <h4 style="text-align:left; margin-top:5px;">
                Department of Electronics and Communication Engineering
            </h4>
            <h3 style="text-align:left; margin-top:10px;">
                AQVH FINAL: Advanced BB84 Quantum Key Distribution Simulator
            </h3>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")

    # session state animation (SUPER FIXED)
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = False
    if "anim_s" not in st.session_state:
        st.session_state.anim_s = 0
    if "animation_shown" not in st.session_state:
        st.session_state.animation_shown = False

    sim = BB84Simulator()

    # BB84 Protocol Description
    st.header("📚 BB84 Quantum Key Distribution Process")
    st.markdown("""
    **BB84** is the first quantum key distribution protocol, proposed by Charles Bennett and Gilles Brassard in 1984. It allows two parties, Alice and Bob, to securely share a secret key over an insecure channel, with security guaranteed by quantum mechanics.

    ### Steps of BB84:
    1. **Key Generation by Alice:**
       - Alice generates a random sequence of bits (0 or 1).
       - For each bit, she randomly chooses a basis: Rectilinear (Z) or Diagonal (X).
       - She prepares qubits accordingly:
         - Z basis: |0⟩ or |1⟩
         - X basis: |+⟩ = (|0⟩+|1⟩)/√2 or |-⟩ = (|0⟩-|1⟩)/√2
       - Sends the qubits to Bob over a quantum channel.

    2. **Measurement by Bob:**
       - Bob randomly chooses a basis (Z or X) for each qubit.
       - Measures the qubit in his chosen basis.
       - Records the measurement result (0 or 1).

    3. **Basis Announcement:**
       - Alice and Bob publicly announce their chosen bases (not the bits).
       - They keep only the bits where bases matched (sifted key).
       - Discard bits where bases differed.

    4. **Error Estimation:**
       - Alice and Bob publicly compare a subset of the sifted key to estimate the Quantum Bit Error Rate (QBER).
       - If QBER is below a threshold (e.g., 11%), proceed; else, abort.

    5. **Privacy Amplification:**
       - Use hashing to distill a shorter, secure key from the sifted key.
       - This removes information that might have leaked to an eavesdropper.

    ### Security:
    - Any eavesdropping (Eve intercepting qubits) introduces errors detectable by QBER.
    - Quantum no-cloning theorem prevents perfect copying without detection.

    ### This Simulator:
    - Simulates the above process with/without Eve.
    - Includes channel noise.
    - Visualizes timelines, Bloch spheres, and generates reports.
    """)

    # Simulation Controls
    with st.sidebar:
        st.header("🎯 Simulation Controls")
        num_bits = st.slider("Qubits to transmit", 50, 2000, 50, step=50)
        threshold = st.slider("QBER Threshold", 0.00, 0.25, 0.11, step=0.01)
        eve_prob = st.slider("Eve Interception Probability", 0.0, 1.0, 0.5, step=0.05)
        eve_attack = st.selectbox("Eve Attack Type", ["Intercept-Resend"], index=0)
        noise_prob = st.slider("Channel Noise Probability", 0.0, 0.1, 0.01, step=0.005)
        window = st.slider("Timeline Window Size", 30, 250, 80, step=10)
        speed = st.slider("Animation Speed", 0.02, 0.30, 0.08, step=0.01)
        pdf_max = st.slider("PDF-style timeline max bits", 20, 200, 50, step=10)
        sifted_display_size = st.slider("Sifted Key Display Size", 10, 200, 20, step=10)

    # Main area start button
    col1, col2 = st.columns([3,1])
    with col2:
        start_sim = st.button("**Start Simulation**", type="primary")

    if start_sim:
        with st.spinner("🔬 **Running Advanced BB84 Quantum Simulation...**"):
            progress_bar = st.progress(0)
            progress_bar.progress(25, text="Initializing quantum simulator...")

            alice_bits = np.random.randint(0, 2, num_bits)
            alice_bases = np.random.randint(0, 2, num_bits)
            bob_bases = np.random.randint(0, 2, num_bits)

            progress_bar.progress(50, text="Simulating quantum transmission...")
            bob_no_eve, eve_results_no = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=False, noise_prob=noise_prob)
            bob_eve, eve_results_eve = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=True, eve_intercept_prob=eve_prob, noise_prob=noise_prob)

            progress_bar.progress(75, text="Analyzing results and generating reports...")
            def compute(t_bob):
                timeline = create_transmission_timeline(alice_bits, alice_bases, bob_bases, t_bob)
                used = timeline[timeline["Used"] == True]
                errors = int(((used["Error"] == True)).sum())
                qber = errors / len(used) if len(used) > 0 else 0.0
                sec = sim.assess_security(float(qber), float(threshold))
                status = sec['status']

                sifted_key = used["AliceBit"].astype(int).tolist()
                final_key = sim.privacy_amplification(sifted_key, qber) if status == "SECURE" else []

                return timeline, errors, qber, status, len(used), len(final_key)

            tl_no_eve, err_no, qber_no, stat_no, sift_no, key_no = compute(bob_no_eve)
            tl_eve, err_e, qber_e, stat_e, sift_e, key_e = compute(bob_eve)

            progress_bar.progress(100, text="Simulation complete!")
            progress_bar.empty()


        with st.spinner("🔬 **Running BB84 Quantum Simulation...**"):
            progress_bar = st.progress(0)
            progress_bar.progress(25, text="Initializing quantum simulator...")

            alice_bits = np.random.randint(0, 2, num_bits)
            alice_bases = np.random.randint(0, 2, num_bits)
            bob_bases = np.random.randint(0, 2, num_bits)

            progress_bar.progress(50, text="Simulating quantum transmission...")
            bob_no_eve, eve_results_no = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=False, noise_prob=noise_prob)
            bob_eve, eve_results_eve = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=True, eve_intercept_prob=eve_prob, noise_prob=noise_prob)

            progress_bar.progress(75, text="Analyzing results and generating reports...")
            def compute(t_bob):
                timeline = create_transmission_timeline(alice_bits, alice_bases, bob_bases, t_bob)
                used = timeline[timeline["Used"] == True]
                errors = int(((used["Error"] == True)).sum())
                qber = errors / len(used) if len(used) > 0 else 0.0
                sec = sim.assess_security(float(qber), float(threshold))
                status = sec['status']

                sifted_key = used["AliceBit"].astype(int).tolist()
                final_key = sim.privacy_amplification(sifted_key, qber) if status == "SECURE" else []

                return timeline, errors, qber, status, len(used), len(final_key)

            tl_no_eve, err_no, qber_no, stat_no, sift_no, key_no = compute(bob_no_eve)
            tl_eve, err_e, qber_e, stat_e, sift_e, key_e = compute(bob_eve)

            progress_bar.progress(100, text="Simulation complete!")
            progress_bar.empty()

            st.success("✅ **Simulation completed successfully!**")
            st.balloons()

        # BB84 Process Animation
        st.markdown("### **BB84 Process Animation**")

        anim_scenario = st.selectbox("**Select Scenario for Animation:**", ["Without Eve", "With Eve"], index=0)
        anim_speed = st.slider("**Animation Speed (seconds per step):**", 0.5, 3.0, 1.0, step=0.1)

        if st.button("**Start BB84 Animation**", type="primary") or st.session_state.animation_shown:
            st.session_state.animation_shown = True

            progress_bar = st.progress(0)
            anim_placeholder = st.empty()

            steps = [
                ("**Step 1: Alice Generates Random Bits**", "Alice creates random bits and bases for secure communication.", str(alice_bits[:20]) + "..."),
                ("**Step 2: Alice Prepares Qubits**", "Alice encodes bits into quantum states using chosen bases.", str([state_label(b, a) for b, a in zip(alice_bits[:20], alice_bases[:20])]) + "..."),
                ("**Step 3: Qubits Sent to Bob**", "Qubits are transmitted through the quantum channel.", "Transmission in progress..."),
                ("**Step 4: Bob Measures Qubits**", "Bob measures qubits in his randomly chosen bases.", str(bob_no_eve[:20] if anim_scenario == "Without Eve" else bob_eve[:20]) + "..."),
                ("**Step 5: Basis Announcement**", "Alice and Bob publicly announce their bases (not the bits).", "Bases compared..."),
                ("**Step 6: Sifting**", "Only bits with matching bases are kept (sifted key).", str(tl_no_eve[tl_no_eve["Used"] == True]["AliceBit"].tolist()[:20] if anim_scenario == "Without Eve" else tl_eve[tl_eve["Used"] == True]["AliceBit"].tolist()[:20]) + "..."),
                ("**Step 7: Error Estimation**", f"QBER calculated to detect eavesdropping: {qber_no:.3f}" if anim_scenario == "Without Eve" else f"QBER: {qber_e:.3f}", "Checking for eavesdropping..."),
                ("**Step 8: Privacy Amplification**", "Hashing reduces key length for security.", "Key generated securely." if (key_no > 0 if anim_scenario == "Without Eve" else key_e > 0) else "No secure key due to errors.")
            ]

            full_content = ""
            for i, (title, desc, data) in enumerate(steps):
                progress_bar.progress((i+1) / len(steps))
                step_html = f"""
                <div style="border: 2px solid #0f62fe; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f0f8ff;">
                    <h3 style="color: #0f62fe;">{title}</h3>
                    <p>{desc}</p>
                    <pre style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; font-family: monospace;">{data}</pre>
                </div>
                """
                full_content += step_html

            anim_placeholder.markdown(full_content, unsafe_allow_html=True)
            progress_bar.empty()
            st.markdown("### ✅ **Animation Complete!**")

            # Reset button
            if st.button("**Reset Animation**"):
                st.session_state.animation_shown = False
                st.rerun()


            # Action buttons row
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            with action_col1:
                if st.button("**Detailed Metrics**", help="Show comprehensive performance metrics"):
                    st.session_state.show_detailed = not st.session_state.get('show_detailed', False)
            with action_col2:
                if st.button("**Error Analysis**", help="Analyze error patterns in detail"):
                    st.session_state.show_errors = not st.session_state.get('show_errors', False)
            with action_col3:
                if st.button("**Key Statistics**", help="Display key generation statistics"):
                    st.session_state.show_stats = not st.session_state.get('show_stats', False)
            with action_col4:
                if st.button("**Parameter Summary**", help="Show current simulation parameters"):
                    st.session_state.show_params = not st.session_state.get('show_params', False)

        c1, c2 = st.columns(2)

        with c1:
                st.subheader("✅ **No Eavesdropper Scenario**")
                st.metric("📡 Transmitted Qubits", num_bits)
                st.metric("🔗 Sifted Bits", sift_no)
                st.metric("❌ Errors Detected", err_no)
                st.metric("📊 Quantum Bit Error Rate", f"{qber_no:.4f}")
                st.metric("🔐 Final Secure Key", key_no)
                st.metric("⚡ Key Generation Rate", f"{key_no / num_bits:.4f}" if num_bits > 0 else "0")
                st.plotly_chart(qber_gauge(qber_no, threshold), use_container_width=True, key="gauge_no")

        with c2:
                st.subheader("🕵️ **Eavesdropper Present Scenario**")
                st.metric("📡 Transmitted Qubits", num_bits)
                st.metric("🔗 Sifted Bits", sift_e)
                st.metric("❌ Errors Detected", err_e)
                st.metric("📊 Quantum Bit Error Rate", f"{qber_e:.4f}")
                st.metric("🔐 Final Secure Key", key_e)
                st.metric("⚡ Key Generation Rate", f"{key_e / num_bits:.4f}" if num_bits > 0 else "0")
                st.plotly_chart(qber_gauge(qber_e, threshold), use_container_width=True, key="gauge_e")

            # Conditional displays
        if st.session_state.get('show_detailed', False):
                st.markdown("### 📊 **Detailed Performance Metrics**")
                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.markdown("**No Eve:**")
                    st.info(f"• Efficiency: {sift_no/num_bits:.1%}\n• Security: {stat_no}\n• Key Rate: {key_no/num_bits:.3f}")
                with det_col2:
                    st.markdown("**With Eve:**")
                    st.info(f"• Efficiency: {sift_e/num_bits:.1%}\n• Security: {stat_e}\n• Key Rate: {key_e/num_bits:.3f}")

        if st.session_state.get('show_errors', False):
            st.markdown("### 🔍 **Error Pattern Analysis**")
            err_col1, err_col2 = st.columns(2)
            with err_col1:
                st.markdown("**No Eve Error Distribution:**")
                if err_no > 0:
                    st.warning(f"Errors found at positions: {tl_no_eve[tl_no_eve['Error']==True]['BitIndex'].tolist()[:10]}...")
                else:
                    st.success("No errors detected!")
            with err_col2:
                st.markdown("**With Eve Error Distribution:**")
                if err_e > 0:
                    st.error(f"Errors found at positions: {tl_eve[tl_eve['Error']==True]['BitIndex'].tolist()[:10]}...")
                else:
                    st.info("Unexpected: No errors with Eve present")

        if st.session_state.get('show_stats', False):
            st.markdown("### 📈 **Key Generation Statistics**")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("**Raw Key Length**", max(key_no, key_e))
            with stat_col2:
                st.metric("**Privacy Amplification**", f"{max(key_no, key_e)/max(sift_no, sift_e):.2f}x" if max(sift_no, sift_e) > 0 else "N/A")
            with stat_col3:
                st.metric("**Security Level**", f"2^(-{int(np.log2(1e-6))})")
        if st.session_state.get('show_params', False):
            st.markdown("### ⚙️ **Simulation Parameters**")
            params = {
                "Qubits Transmitted": num_bits,
                "QBER Threshold": f"{threshold:.2f}",
                "Eve Intercept Probability": f"{eve_prob:.2f}",
                "Channel Noise Probability": f"{noise_prob:.4f}",
                "Eve Attack Type": eve_attack,
                "Timeline Window Size": window,
                "Animation Speed": f"{speed:.2f}s"
            }
            st.json(params)
            # QBER Analysis Graph (from bb84_app2.py)
        if sift_no > 0 or sift_e > 0:
            st.markdown("### 📊 **QBER Analysis**")
            fig, ax = plt.subplots(figsize=(10, 5))
            # For No Eve
            if sift_no > 0:
                window_size = max(1, sift_no // 10)
                local_qbers_no = []
                for i in range(0, sift_no, window_size):
                    a_w = tl_no_eve[tl_no_eve["Used"] == True]["AliceBit"].iloc[i:i + window_size].values
                    b_w = tl_no_eve[tl_no_eve["Used"] == True]["BobResult"].iloc[i:i + window_size].values
                    if len(a_w) > 0:
                        local_qbers_no.append(float(np.mean(a_w != b_w)))
                ax.plot(local_qbers_no, 'bo-', label=f'No Eve Local QBER (Overall: {qber_no:.3f})')

            # For With Eve
            if sift_e > 0:
                window_size = max(1, sift_e // 10)
                local_qbers_e = []
                for i in range(0, sift_e, window_size):
                    a_w = tl_eve[tl_eve["Used"] == True]["AliceBit"].iloc[i:i + window_size].values
                    b_w = tl_eve[tl_eve["Used"] == True]["BobResult"].iloc[i:i + window_size].values
                    if len(a_w) > 0:
                        local_qbers_e.append(float(np.mean(a_w != b_w)))
                ax.plot(local_qbers_e, 'ro-', label=f'With Eve Local QBER (Overall: {qber_e:.3f})')

            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {threshold}')
            ax.set_xlabel('Window Index')
            ax.set_ylabel('Error Rate')
            ax.set_title('Quantum Bit Error Rate (QBER) Analysis')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Sifted Bits Table (from bb84_app2.py)
            st.markdown("### 📋 **Sifted Bits Comparison**")
            col_no, col_e = st.columns(2)
            with col_no:
                st.markdown("**No Eve (First {} Sifted Bits):**".format(min(sifted_display_size, sift_no)))
                if sift_no > 0:
                    show_n = min(sifted_display_size, sift_no)
                    df_no = pd.DataFrame({
                        "Alice": tl_no_eve[tl_no_eve["Used"] == True]["AliceBit"].iloc[:show_n].values,
                        "Bob": tl_no_eve[tl_no_eve["Used"] == True]["BobResult"].iloc[:show_n].values,
                        "Match": tl_no_eve[tl_no_eve["Used"] == True]["Error"].iloc[:show_n].apply(lambda x: not x).values
                    })
                    st.dataframe(df_no)
                else:
                    st.info("No sifted bits.")

            with col_e:
                st.markdown("**With Eve (First {} Sifted Bits):**".format(min(sifted_display_size, sift_e)))
                if sift_e > 0:
                    show_n = min(sifted_display_size, sift_e)
                    df_e = pd.DataFrame({
                        "Alice": tl_eve[tl_eve["Used"] == True]["AliceBit"].iloc[:show_n].values,
                        "Bob": tl_eve[tl_eve["Used"] == True]["BobResult"].iloc[:show_n].values,
                        "Match": tl_eve[tl_eve["Used"] == True]["Error"].iloc[:show_n].apply(lambda x: not x).values
                    })
                    st.dataframe(df_e)
                else:
                    st.info("No sifted bits.")

            st.plotly_chart(decision_line(qber_e, threshold, "**Attack Detection Decision Analysis**"),
                            use_container_width=True, key="dec_line")

        # Create tabs for additional analysis
        tab2, tab3, tab4, tab5, tab6 = st.tabs(["Timeline Analysis", "Comparative Analysis", "Quantum Visualization", "Report Generation", "Protocol Guide"])

        # ------------------ Timeline ------------------
        with tab2:
            st.markdown("### 📈 ** Timeline Analysis**")

            # Scenario selection with radio button
            mode = st.radio("**Select Scenario:**", ["Without Eve", "With Eve"],
                          horizontal=True, index=0 if 'mode' not in st.session_state else (0 if st.session_state.mode == "Without Eve" else 1))
            st.session_state.mode = mode

            TL = tl_no_eve if mode == "Without Eve" else tl_eve

            # Visualization options
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            with viz_col1:
                show_pdf = st.checkbox("📄 **PDF Style Timeline**", value=True)
            with viz_col2:
                show_plotly = st.checkbox("📊 **Interactive Plotly**", value=True)
            with viz_col3:
                show_inspector = st.checkbox("🔍 **Qubit Inspector**", value=True)
            fig_pdf = plot_pdf_style_timeline(TL, title=f"{mode} Scenario", max_bits=pdf_max)
            st.pyplot(fig_pdf)

            st.markdown("---")
            st.subheader("✅ Plotly Timeline (Self Interactive)")
            start, end = st.slider("Select range", 0, len(TL) - 1, (0, min(len(TL) - 1, window)),
                                   key=f"range_{mode}")

            st.plotly_chart(
                plotly_bit_timeline(TL, start, end, title=f"{mode} - Plotly Timeline"),
                use_container_width=True,
                key=f"plotly_timeline_{mode}"
            )
            st.plotly_chart(
                plotly_error_timeline(TL, start, end, title=f"{mode} - Error Timeline"),
                use_container_width=True,
                key=f"plotly_err_{mode}"
            )

            st.markdown("---")
            st.subheader("🔍 Inspect Individual Qubit")
            inspect_idx = st.slider("Select Qubit Index to Inspect", start, end, start, key=f"inspect_{mode}")
            bit = int(alice_bits[inspect_idx])
            basis = int(alice_bases[inspect_idx])
            bob_bit = int(TL.loc[inspect_idx, "BobResult"])
            match = TL.loc[inspect_idx, "BaseMatch"]
            error = TL.loc[inspect_idx, "Error"]

            st.markdown(f"""
            **Index:** {inspect_idx}
            **Alice Bit:** {bit}, **Basis:** {basis} ({'Z' if basis==0 else 'X'})
            **Bob Basis:** {int(TL.loc[inspect_idx, 'BobBasis'])}, **Bob Bit:** {bob_bit}
            **Basis Match:** {'Yes' if match else 'No'}, **Error:** {'Yes' if error else 'No'}
            **State:** {state_label(bit, basis)}
            """)

            try:
                sv = get_statevector_from_bit_basis(bit, basis)
                st.pyplot(plot_bloch_multivector(sv))
            except:
                st.error("Bloch failed")

            st.markdown("---")
            st.subheader("� **Advanced Timeline Animation Controls**")

            anim_col1, anim_col2, anim_col3, anim_col4 = st.columns(4)

            with anim_col1:
                if st.button("**Start Animation**",
                           help="Begin animated timeline visualization",
                           key=f"anim_start_{mode}"):
                    st.session_state.anim_running = True

            with anim_col2:
                if st.button("**Pause Animation**",
                           help="Pause the current animation",
                           key=f"anim_pause_{mode}"):
                    st.session_state.anim_running = False

            with anim_col3:
                if st.button("**Stop & Reset**",
                           help="Stop animation and reset to beginning",
                           key=f"anim_reset_{mode}"):
                    st.session_state.anim_running = False
                    st.session_state.anim_s = 0

            with anim_col4:
                if st.button("**Fast Forward**",
                           help="Skip to end of timeline",
                           key=f"anim_skip_{mode}"):
                    st.session_state.anim_running = False
                    st.session_state.anim_s = num_bits - window

            anim_box = st.empty()

            if st.session_state.anim_running:
                step = max(10, window // 3)
                for s in range(st.session_state.anim_s, num_bits, step):
                    if not st.session_state.anim_running:
                        st.session_state.anim_s = s
                        break

                    e = min(num_bits - 1, s + window)
                    anim_box.plotly_chart(
                        plotly_bit_timeline(TL, s, e, title=f"Animated Window: {s} → {e} ({mode})"),
                        use_container_width=True,
                        key=f"anim_{mode}_{s}_{e}"
                    )
                    time.sleep(speed)

                if st.session_state.anim_running:
                    st.session_state.anim_running = False
                    st.session_state.anim_s = 0

        # ------------------ Comparison PDF style ------------------
        with tab3:
            st.subheader("🆚 **Comparative Analysis: No Eve vs With Eve**")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.bar(["Transmitted", "Sifted", "Final Key"],
                    [num_bits, sift_no, key_no], color=['skyblue', 'lightgreen', 'gold'])
            ax1.set_title(f"No Eve (QBER: {qber_no:.3f})")
            ax1.set_ylabel("Number of Bits")

            ax2.bar(["Transmitted", "Sifted", "Final Key"],
                    [num_bits, sift_e, key_e], color=['skyblue', 'salmon', 'darkred'])
            ax2.set_title(f"With Eve (QBER: {qber_e:.3f})")
            ax2.set_ylabel("Number of Bits")

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🔑 **Generated Keys**")
            key_col1, key_col2 = st.columns(2)
            with key_col1:
                st.markdown("**No Eve Key:**")
                if key_no > 0:
                    key_no_str = ''.join(map(str, sim.privacy_amplification(tl_no_eve[tl_no_eve["Used"] == True]["AliceBit"].astype(int).tolist(), qber_no)))
                    st.code(key_no_str[:100] + "..." if len(key_no_str) > 100 else key_no_str, language="text")
                else:
                    st.warning("No secure key generated.")

            with key_col2:
                st.markdown("**With Eve Key:**")
                if key_e > 0:
                    key_e_str = ''.join(map(str, sim.privacy_amplification(tl_eve[tl_eve["Used"] == True]["AliceBit"].astype(int).tolist(), qber_e)))
                    st.code(key_e_str[:100] + "..." if len(key_e_str) > 100 else key_e_str, language="text")
                else:
                    st.error("No secure key generated due to eavesdropping.")

            # Download buttons for keys
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if key_no > 0:
                    key_no_bytes = bytes(key_no_str, 'utf-8')  # Assuming key_no_str is the full key
                    st.download_button(
                        label="**Download No Eve Key**",
                        data=key_no_bytes,
                        file_name="bb84_no_eve_key.txt",
                        mime="text/plain",
                        help="Download the secure key for No Eve scenario"
                    )
            with dl_col2:
                if key_e > 0:
                    key_e_bytes = bytes(key_e_str, 'utf-8')
                    st.download_button(
                        label="**Download With Eve Key**",
                        data=key_e_bytes,
                        file_name="bb84_with_eve_key.txt",
                        mime="text/plain",
                        help="Download the secure key for With Eve scenario"
                    )

            st.markdown("### Key Observation")
            if qber_e > threshold:
                st.error("🕵️ Eve detected: QBER exceeds threshold → Abort key exchange ✅")
            else:
                st.success("🔒 Channel secure: QBER below threshold ✅")

        # ------------------ Bloch sphere ADVANCED ------------------
        with tab4:
            st.markdown("### 🧿 **Advanced Quantum State Visualization**")

            # Mode selection buttons
            mode_buttons_col1, mode_buttons_col2 = st.columns(2)
            with mode_buttons_col1:
                single_mode = st.button("🔍 **Single Qubit Analysis**", help="Analyze individual quantum states")
            with mode_buttons_col2:
                range_mode = st.button("📊 **Multi-Qubit Comparison**", help="Compare multiple quantum states")

            if single_mode or not range_mode:
                st.markdown("---")
                st.subheader("🔍 **Single Qubit Quantum State Analysis**")
                idx = st.slider("**Select Qubit Index**", 0, len(alice_bits) - 1, 0, key="bloch_idx")
                bit = int(alice_bits[idx])
                basis = int(alice_bases[idx])

                # Enhanced display
                state_col1, state_col2 = st.columns([1, 2])
                with state_col1:
                    st.markdown(f"""
                    **🎯 Selected Qubit #{idx}:**
                    - **Bit Value:** `{bit}` ({'🔵 |0⟩' if bit == 0 else '🔴 |1⟩'})
                    - **Basis:** `{basis}` ({'➕ Z-Basis (Rectilinear)' if basis == 0 else '➖ X-Basis (Diagonal)'})
                    - **Quantum State:** `{state_label(bit, basis)}`
                    - **State Vector:** |ψ⟩ = {'|0⟩' if bit == 0 and basis == 0 else '|1⟩' if bit == 1 and basis == 0 else '|+⟩' if bit == 0 and basis == 1 else '|-⟩'}
                    """)

                with state_col2:
                    try:
                        sv = get_statevector_from_bit_basis(bit, basis)
                        st.markdown("**🧿 Bloch Sphere Representation:**")
                        # Use Plotly Bloch sphere for better interactivity
                        fig = plotly_bloch_sphere([sv])
                        st.plotly_chart(fig, use_container_width=True)

                        # Add state vector display
                        st.markdown("**📐 State Vector Details:**")
                        st.code(f"Statevector: {sv}")

                    except Exception as e:
                        st.error(f"❌ Bloch sphere visualization failed: {e}")
                        st.info("💡 Make sure Qiskit is properly installed")
            else:
                st.markdown("---")
                st.subheader("📊 **Multi-Qubit Quantum State Comparison**")
                start_idx, end_idx = st.slider("**Select Qubit Range**", 0, len(alice_bits) - 1, (0, min(10, len(alice_bits)-1)), key="bloch_range")

                st.markdown(f"**Comparing qubits {start_idx} to {end_idx} ({end_idx - start_idx + 1} qubits)**")

                states = []
                state_info = []
                for i in range(start_idx, end_idx + 1):
                    bit = int(alice_bits[i])
                    basis = int(alice_bases[i])
                    sv = get_statevector_from_bit_basis(bit, basis)
                    states.append(sv)
                    state_info.append(f"Qubit {i}: {state_label(bit, basis)}")

                # Display state information
                st.markdown("**📋 Quantum States in Range:**")
                for info in state_info:
                    st.markdown(f"• {info}")

                try:
                    st.markdown("**🌐 3D Bloch Sphere Multi-State View:**")
                    st.plotly_chart(plotly_bloch_sphere(states), use_container_width=True)
                    st.info("💡 Each point represents a quantum state. Hover for details.")
                except Exception as e:
                    st.error(f"❌ Interactive Bloch sphere failed: {e}")
                    st.info("💡 Falling back to individual visualizations...")

                    # Fallback to individual plots
                    fallback_cols = st.columns(min(3, len(states)))
                    for i, sv in enumerate(states):
                        with fallback_cols[i % len(fallback_cols)]:
                            st.pyplot(plot_bloch_multivector(sv))
                            st.caption(f"Qubit {start_idx + i}")

        # ------------------ Report ------------------
        with tab5:
            st.markdown("### 📄 **Professional Report Generation**")

            # Report options
            report_col1, report_col2, report_col3 = st.columns(3)
            with report_col1:
                include_charts = st.checkbox("📊 **Include Charts**", value=True)
            with report_col2:
                include_timeline = st.checkbox("📈 **Include Timeline**", value=True)
            with report_col3:
                detailed_analysis = st.checkbox("🔬 **Detailed Analysis**", value=True)

            st.markdown("---")
            st.subheader("📋 **Report Downloads**")

            project_info = {
                "University": "JNTUACEA",
                "Department": "ECE",
                "Project": "AQVH FINAL: Advanced BB84 QKD Simulator",
                "Generated On": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Total Qubits": num_bits,
                "Eve Interception Probability": eve_prob,
                "Eve Attack Type": eve_attack,
                "Channel Noise Probability": noise_prob,
                "QBER Threshold": threshold
            }

            summary = {
                "No Eve -> QBER": f"{qber_no:.4f}",
                "No Eve -> Errors": err_no,
                "No Eve -> Final Key Length": key_no,
                "With Eve -> QBER": f"{qber_e:.4f}",
                "With Eve -> Errors": err_e,
                "With Eve -> Final Key Length": key_e
            }

            # Download buttons with enhanced styling
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    "📊 **CSV: No Eve Data**",
                    data=tl_no_eve.to_csv(index=False).encode("utf-8"),
                    file_name="AQVH_No_Eve_Timeline.csv",
                    mime="text/csv",
                    help="Download detailed timeline data for secure channel"
                )
            with dl_col2:
                st.download_button(
                    "🕵️ **CSV: With Eve Data**",
                    data=tl_eve.to_csv(index=False).encode("utf-8"),
                    file_name="AQVH_With_Eve_Timeline.csv",
                    mime="text/csv",
                    help="Download detailed timeline data for compromised channel"
                )
            with dl_col3:
                pdf_bytes = create_pdf_report_with_graphs(
                    project_info=project_info,
                    summary=summary,
                    timeline_df_no_eve=tl_no_eve,
                    timeline_df_eve=tl_eve,
                    num_bits=num_bits,
                    sift_no=sift_no, key_no=key_no, qber_no=qber_no,
                    sift_e=sift_e, key_e=key_e, qber_e=qber_e,
                    threshold=threshold,
                    pdf_max_bits=pdf_max
                )
                st.download_button(
                    "📄 **PDF Full Report**",
                    data=pdf_bytes,
                    file_name="AQVH_FINAL_BB84_Report.pdf",
                    mime="application/pdf",
                    help="Download comprehensive PDF report with all analysis"
                )

        # ------------------ BB84 Process Guide ------------------
        with tab6:
            st.header("📚 BB84 Quantum Key Distribution Process")
            st.markdown("""
            **BB84** is the first quantum key distribution protocol, proposed by Charles Bennett and Gilles Brassard in 1984. It allows two parties, Alice and Bob, to securely share a secret key over an insecure channel, with security guaranteed by quantum mechanics.

            ### Steps of BB84:
            1. **Key Generation by Alice:**
               - Alice generates a random sequence of bits (0 or 1).
               - For each bit, she randomly chooses a basis: Rectilinear (Z) or Diagonal (X).
               - She prepares qubits accordingly:
                 - Z basis: |0⟩ or |1⟩
                 - X basis: |+⟩ = (|0⟩+|1⟩)/√2 or |-⟩ = (|0⟩-|1⟩)/√2
               - Sends the qubits to Bob over a quantum channel.

            2. **Measurement by Bob:**
               - Bob randomly chooses a basis (Z or X) for each qubit.
               - Measures the qubit in his chosen basis.
               - Records the measurement result (0 or 1).

            3. **Basis Announcement:**
               - Alice and Bob publicly announce their chosen bases (not the bits).
               - They keep only the bits where bases matched (sifted key).
               - Discard bits where bases differed.

            4. **Error Estimation:**
               - Alice and Bob publicly compare a subset of the sifted key to estimate the Quantum Bit Error Rate (QBER).
               - If QBER is below a threshold (e.g., 11%), proceed; else, abort.

            5. **Privacy Amplification:**
               - Use hashing to distill a shorter, secure key from the sifted key.
               - This removes information that might have leaked to an eavesdropper.

            ### Security:
            - Any eavesdropping (Eve intercepting qubits) introduces errors detectable by QBER.
            - Quantum no-cloning theorem prevents perfect copying without detection.

            ### This Simulator:
            - Simulates the above process with/without Eve.
            - Includes channel noise.
            - Visualizes timelines, Bloch spheres, and generates reports.
            """)

    else:
        st.info("Click **Run Simulation** to generate results.")


if __name__ == "__main__":
    main()
