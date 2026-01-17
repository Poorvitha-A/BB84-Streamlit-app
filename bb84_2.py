# ============================================================
# BB84 Quantum Key Distribution Simulator - Main Application
# University: Jawaharlal Nehru Technological University Anantapur
# Department: Electronics and Communication Engineering
# Project: AQVH FINAL - BB84 QKD Simulator
# ============================================================

# ============================================================
# IMPORTS - ORGANIZED BY CATEGORY
# ============================================================

# Standard Library Imports
import io
import hashlib
import time
from datetime import datetime

# Third-party Imports - Data & Scientific Computing
import numpy as np
import pandas as pd

# Third-party Imports - Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch

# Third-party Imports - Quantum Computing (Qiskit)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit_aer import AerSimulator
from qiskit import transpile

# Streamlit - Web UI Framework
import streamlit as st

# Local Modules - Configuration & Utilities
import bb84_config as config
from bb84_simulator import BB84Simulator
from bb84_utils import (
    create_transmission_timeline,
    compute_metrics,
    analyze_error_patterns,
    calculate_key_rate,
    get_basis_distribution,
    get_bit_distribution,
    calculate_eve_impact
)
from bb84_visualizations import (
    plot_pdf_style_timeline,
    plotly_bit_timeline,
    plotly_error_timeline,
    qber_gauge,
    decision_line,
    plotly_bloch_sphere,
)

# ============================================================
# BLOCH SPHERE FRAGMENT - Isolated slider updates
# ============================================================

@st.fragment
def render_single_qubit_bloch():
    """Render single qubit Bloch sphere in isolated fragment"""
    st.subheader("🔍 **Single Qubit Quantum State Analysis**")
    idx = st.slider(
        "**Select Qubit Index**", 
        0, 
        len(st.session_state.alice_bits_stored) - 1, 
        value=st.session_state.bloch_single_idx,
        key="bloch_single_idx"
    )
    bit = int(st.session_state.alice_bits_stored[idx])
    basis = int(st.session_state.alice_bases_stored[idx])

    state_col1, state_col2 = st.columns([1, 2])
    with state_col1:
        st.markdown(f"""
        **🎯 Selected Qubit #{idx}:**
        - **Bit Value:** `{bit}` ({'🔵 |0⟩' if bit == 0 else '🔴 |1⟩'})
        - **Basis:** `{basis}` ({'➕ Z-Basis (Rectilinear)' if basis == 0 else '➖ X-Basis (Diagonal)'})
        - **Quantum State:** `{BB84Simulator.state_label(bit, basis)}`
        - **State Vector:** |ψ⟩ = {'|0⟩' if bit == 0 and basis == 0 else '|1⟩' if bit == 1 and basis == 0 else '|+⟩' if bit == 0 and basis == 1 else '|-⟩'}
        """)

    with state_col2:
        try:
            sv = BB84Simulator.get_statevector_from_bit_basis(bit, basis)
            st.markdown("**🧿 Bloch Sphere Representation:**")
            st.markdown("""
            **What the Bloch Sphere Shows:**
            - **Position on sphere**: Represents the quantum state of the qubit
            - **North Pole (Z=1)**: |0⟩ state (classical bit 0)
            - **South Pole (Z=-1)**: |1⟩ state (classical bit 1)
            - **Equatorial plane**: Superposition states (| + ⟩, | - ⟩)
            - **X-axis**: Real part of superposition amplitude
            - **Y-axis**: Imaginary part of superposition amplitude
            - **Z-axis**: Probability difference between |0⟩ and |1⟩
            """)
            fig = plotly_bloch_sphere([sv])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**📐 State Vector Details:**")
            st.code(f"Statevector: {sv}")

        except Exception as e:
            st.error(f"❌ Bloch sphere visualization failed: {e}")
            st.info("💡 Make sure Qiskit is properly installed")


@st.fragment
def render_multi_qubit_bloch():
    """Render multi-qubit Bloch sphere in isolated fragment"""
    st.subheader("📊 **Multi-Qubit Quantum State Comparison**")
    bloch_range = st.slider(
        "**Select Qubit Range**", 
        0, 
        len(st.session_state.alice_bits_stored) - 1, 
        value=(st.session_state.bloch_range_start, st.session_state.bloch_range_end),
        key="bloch_range_key"
    )
    start_idx, end_idx = bloch_range

    st.markdown(f"**Comparing qubits {start_idx} to {end_idx} ({end_idx - start_idx + 1} qubits)**")

    states = []
    state_info = []
    for i in range(start_idx, end_idx + 1):
        bit = int(st.session_state.alice_bits_stored[i])
        basis = int(st.session_state.alice_bases_stored[i])
        sv = BB84Simulator.get_statevector_from_bit_basis(bit, basis)
        states.append(sv)
        state_info.append(f"Qubit {i}: {BB84Simulator.state_label(bit, basis)}")

    st.markdown("**📋 Quantum States in Range:**")
    for info in state_info:
        st.markdown(f"• {info}")

    try:
        st.markdown("**🌐 3D Bloch Sphere Multi-State View:**")
        st.plotly_chart(plotly_bloch_sphere(states), use_container_width=True)
        st.info("💡 Each point represents a quantum state. Hover for details.")
    except Exception as e:
        st.error(f"❌ Multi-qubit Bloch sphere visualization failed: {e}")


@st.fragment
def render_timeline_interactive():
    """Render timeline interactive sliders in isolated fragment"""
    if st.session_state.sim_results is None:
        return
    
    tl_no_eve = st.session_state.sim_results['tl_no_eve']
    tl_eve = st.session_state.sim_results['tl_eve']
    pdf_max = st.session_state.get('pdf_max', config.DEFAULT_PDF_MAX_BITS)
    
    st.markdown("### 📈 **Timeline Analysis**")

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        show_pdf = st.checkbox("📄 **PDF Style Timeline**", value=True)
    with viz_col2:
        show_plotly = st.checkbox("📊 **Interactive Plotly Timeline**", value=True)

    tl_col1, tl_col2 = st.columns(2)
    with tl_col1:
        st.subheader("✅ **No Eavesdropper Scenario**")
        if show_pdf:
            fig_pdf_no = plot_pdf_style_timeline(tl_no_eve, title="No Eve Scenario", max_bits=pdf_max)
            st.pyplot(fig_pdf_no)
        if show_plotly:
            st.markdown("---")
            st.subheader("Plotly Timeline (Interactive)")
            max_no = len(tl_no_eve) - 1
            # Initialize defaults if needed
            if st.session_state.timeline_range_no_end == 0:
                st.session_state.timeline_range_no_end = min(max_no, 100)
            # Get saved values
            saved_start_no = min(st.session_state.timeline_range_no_start, max_no)
            saved_end_no = min(st.session_state.timeline_range_no_end, max_no)
            start_no, end_no = st.slider("Select range", 0, max_no, value=(saved_start_no, saved_end_no), key="range_no_slider_key", on_change=lambda: None)

            st.plotly_chart(
                plotly_bit_timeline(tl_no_eve, start_no, end_no, title="No Eve - Plotly Timeline"),
                use_container_width=True,
                key="plotly_timeline_no"
            )
            st.plotly_chart(
                plotly_error_timeline(tl_no_eve, start_no, end_no, title="No Eve - Error Timeline"),
                use_container_width=True,
                key="plotly_err_no"
            )

    with tl_col2:
        st.subheader("🕵️ **Eavesdropper Present Scenario**")
        if show_pdf:
            fig_pdf_e = plot_pdf_style_timeline(tl_eve, title="With Eve Scenario", max_bits=pdf_max)
            st.pyplot(fig_pdf_e)
        if show_plotly:
            st.markdown("---")
            st.subheader("Plotly Timeline (Interactive)")
            max_e = len(tl_eve) - 1
            # Initialize defaults if needed
            if st.session_state.timeline_range_eve_end == 0:
                st.session_state.timeline_range_eve_end = min(max_e, 100)
            # Get saved values
            saved_start_e = min(st.session_state.timeline_range_eve_start, max_e)
            saved_end_e = min(st.session_state.timeline_range_eve_end, max_e)
            start_e, end_e = st.slider("Select range", 0, max_e, value=(saved_start_e, saved_end_e), key="range_e_slider_key", on_change=lambda: None)

            st.plotly_chart(
                plotly_bit_timeline(tl_eve, start_e, end_e, title="With Eve - Plotly Timeline"),
                use_container_width=True,
                key="plotly_timeline_e"
            )
            st.plotly_chart(
                plotly_error_timeline(tl_eve, start_e, end_e, title="With Eve - Error Timeline"),
                use_container_width=True,
                key="plotly_err_e"
            )


@st.fragment
def render_report_generation():
    """Render report generation in isolated fragment"""
    num_bits = st.session_state.get('num_bits', config.DEFAULT_QUBITS)
    threshold = st.session_state.get('threshold', config.DEFAULT_QBER_THRESHOLD)
    eve_prob = st.session_state.get('eve_prob', config.DEFAULT_EVE_PROB)
    eve_attack = st.session_state.get('eve_attack', 'Intercept-Resend')
    noise_prob = st.session_state.get('noise_prob', config.DEFAULT_NOISE_PROB)
    pdf_max = st.session_state.get('pdf_max', config.DEFAULT_PDF_MAX_BITS)
    
    if st.session_state.sim_results is None:
        return
    
    tl_no_eve = st.session_state.sim_results['tl_no_eve']
    tl_eve = st.session_state.sim_results['tl_eve']
    err_no = st.session_state.sim_results['err_no']
    err_e = st.session_state.sim_results['err_e']
    qber_no = st.session_state.sim_results['qber_no']
    qber_e = st.session_state.sim_results['qber_e']
    sift_no = st.session_state.sim_results['sift_no']
    sift_e = st.session_state.sim_results['sift_e']
    key_no = st.session_state.sim_results['key_no']
    key_e = st.session_state.sim_results['key_e']
    
    st.markdown("### 📄 **Professional Report Generation**")

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
        "University": "JNTUA",
        "College": "JNTUACEA",
        "Department": "ECE",
        "Project": "AQVH FINAL: BB84 QKD Simulator",
        "Team": "Team Silicon",
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
        @st.cache_data
        def get_pdf_bytes():
            project_info_tuple = tuple(sorted(project_info.items()))
            summary_tuple = tuple(sorted(summary.items()))
            timeline_csv_no_eve = tl_no_eve.to_csv(index=False)
            timeline_csv_eve = tl_eve.to_csv(index=False)
            
            return create_pdf_report_with_graphs(
                project_info=project_info_tuple,
                summary=summary_tuple,
                timeline_df_no_eve_csv=timeline_csv_no_eve,
                timeline_df_eve_csv=timeline_csv_eve,
                num_bits=num_bits,
                sift_no=sift_no, key_no=key_no, qber_no=qber_no,
                sift_e=sift_e, key_e=key_e, qber_e=qber_e,
                threshold=threshold,
                pdf_max_bits=pdf_max
            )
        
        pdf_bytes = get_pdf_bytes()
        st.download_button(
            "📄 **PDF Full Report**",
            data=pdf_bytes,
            file_name="AQVH_FINAL_BB84_Report.pdf",
            mime="application/pdf",
            help="Download comprehensive PDF report with all analysis"
        )


# ============================================================
# MAIN APPLICATION - STREAMLIT UI
# ============================================================

def main():
    """Main Streamlit application entry point"""
    st.set_page_config(
        page_title="bb84-qkd-simulator-jntu",
        page_icon="jntua_logo.png",
        layout="wide"
    )
    
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
    
    # Prevent page scrolling on slider changes
    st.markdown("""
    <script>
        window.addEventListener('beforeunload', function() {
            sessionStorage.setItem('scrollPos', window.scrollY);
        });
        window.addEventListener('load', function() {
            const scrollPos = sessionStorage.getItem('scrollPos');
            if (scrollPos) {
                window.scrollTo(0, parseInt(scrollPos));
                sessionStorage.removeItem('scrollPos');
            }
        });
    </script>
    """, unsafe_allow_html=True)

    # ============================================================
    # HEADER SECTION
    # ============================================================
    left, center = st.columns([2, 8])
    with left:
        try:
            st.image("jntua_logo.png", width=200)
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
                bb84-qkd-simulator-jntu: AQVH FINAL BB84 Quantum Key Distribution Simulator
            </h3>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")

    # ============================================================
    # SESSION STATE INITIALIZATION
    # ============================================================
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = False
    if "anim_s" not in st.session_state:
        st.session_state.anim_s = 0
    if "anim_running_no" not in st.session_state:
        st.session_state.anim_running_no = False
    if "anim_s_no" not in st.session_state:
        st.session_state.anim_s_no = 0
    if "anim_running_eve" not in st.session_state:
        st.session_state.anim_running_eve = False
    if "anim_s_eve" not in st.session_state:
        st.session_state.anim_s_eve = 0
    if "viz_mode" not in st.session_state:
        st.session_state.viz_mode = "single"
    if "simulation_run" not in st.session_state:
        st.session_state.simulation_run = False
    if "simulation_completed" not in st.session_state:
        st.session_state.simulation_completed = False
    if "balloons_shown" not in st.session_state:
        st.session_state.balloons_shown = False
    
    # Parameter session state
    if "num_bits" not in st.session_state:
        st.session_state.num_bits = config.DEFAULT_QUBITS
    if "threshold" not in st.session_state:
        st.session_state.threshold = config.DEFAULT_QBER_THRESHOLD
    if "eve_prob" not in st.session_state:
        st.session_state.eve_prob = config.DEFAULT_EVE_PROB
    if "eve_attack" not in st.session_state:
        st.session_state.eve_attack = "Intercept-Resend"
    if "noise_prob" not in st.session_state:
        st.session_state.noise_prob = config.DEFAULT_NOISE_PROB
    if "window" not in st.session_state:
        st.session_state.window = config.DEFAULT_WINDOW_SIZE
    if "speed" not in st.session_state:
        st.session_state.speed = config.DEFAULT_ANIMATION_SPEED
    if "pdf_max" not in st.session_state:
        st.session_state.pdf_max = config.DEFAULT_PDF_MAX_BITS
    if "sifted_display_size" not in st.session_state:
        st.session_state.sifted_display_size = config.DEFAULT_SIFTED_DISPLAY_SIZE
    
    if "last_num_bits" not in st.session_state:
        st.session_state.last_num_bits = None
    
    # Visualization state - initialized early but UI hidden before simulation
    if "bloch_single_idx" not in st.session_state:
        st.session_state.bloch_single_idx = 0
    if "bloch_range_start" not in st.session_state:
        st.session_state.bloch_range_start = 0
    if "bloch_range_end" not in st.session_state:
        st.session_state.bloch_range_end = 10
    if "timeline_range_no_start" not in st.session_state:
        st.session_state.timeline_range_no_start = 0
    if "timeline_range_no_end" not in st.session_state:
        st.session_state.timeline_range_no_end = 0
    if "timeline_range_eve_start" not in st.session_state:
        st.session_state.timeline_range_eve_start = 0
    if "timeline_range_eve_end" not in st.session_state:
        st.session_state.timeline_range_eve_end = 0
    
    # Simulation results storage
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "alice_bits_stored" not in st.session_state:
        st.session_state.alice_bits_stored = None
    if "alice_bases_stored" not in st.session_state:
        st.session_state.alice_bases_stored = None
    if "bob_bases_stored" not in st.session_state:
        st.session_state.bob_bases_stored = None

    # ============================================================
    # INFORMATION SECTION
    # ============================================================
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

    # ============================================================
    # SIMULATION PARAMETERS SECTION
    # ============================================================
    st.header("⚙️ Simulation Parameters")
    st.markdown("**Configure the parameters for the BB84 simulation. Adjust as needed before running.**")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        st.slider("📡 Qubits to transmit", config.MIN_QUBITS, config.MAX_QUBITS, 
                           key="num_bits",
                           step=50,
                           help="Number of qubits Alice will send to Bob")
        st.slider("📊 QBER Threshold", config.MIN_THRESHOLD, config.MAX_THRESHOLD, 
                            key="threshold",
                            step=0.01, 
                            help="Maximum acceptable Quantum Bit Error Rate")
    with param_col2:
        st.slider("🕵️ Eve Interception Probability", 0.0, 1.0, 
                           key="eve_prob",
                           step=0.05, 
                           help="Probability that Eve intercepts each qubit")
        st.slider("📶 Channel Noise Probability", 0.0, 0.1, 
                             key="noise_prob",
                             step=0.005, 
                             help="Probability of bit flip due to channel noise")
    with param_col3:
        eve_attack = st.selectbox("🔧 Eve Attack Type", config.EVE_ATTACK_TYPES, 
                                index=0, 
                                key="select_eve_attack",
                                help="Type of eavesdropping attack")
        st.slider("⚡ Animation Speed", 0.02, 0.30, 
                        key="speed",
                        step=0.01, 
                        help="Speed of the animation in seconds")
        st.slider("📄 PDF-style timeline max bits", 20, 200, 
                          key="pdf_max",
                          step=10, 
                          help="Maximum bits to show in PDF-style timeline")
        st.slider("🔑 Sifted Key Display Size", 10, 200, 
                                      key="sifted_display_size",
                                      step=10, 
                                      help="Number of sifted bits to display in tables")

    # Use values from session state
    num_bits = st.session_state.num_bits
    threshold = st.session_state.threshold
    eve_prob = st.session_state.eve_prob
    noise_prob = st.session_state.noise_prob
    window = st.session_state.window
    speed = st.session_state.speed
    pdf_max = st.session_state.pdf_max
    sifted_display_size = st.session_state.sifted_display_size

    # Display current parameters
    st.markdown("---")
    st.subheader("📋 Current Simulation Parameters")
    params_summary = f"""
    | Parameter | Value |
    |-----------|-------|
    | Qubits to Transmit | {num_bits} |
    | QBER Threshold | {threshold:.2f} |
    | Eve Interception Probability | {eve_prob:.2f} |
    | Channel Noise Probability | {noise_prob:.4f} |
    | Eve Attack Type | {eve_attack} |
    | Animation Speed | {speed:.2f}s |
    | PDF Timeline Max Bits | {pdf_max} |
    | Sifted Key Display Size | {sifted_display_size} |
    """
    st.markdown(params_summary)

    # ============================================================
    # SIMULATION EXECUTION SECTION
    # ============================================================
    st.markdown("---")
    run_button = st.button("**Run BB84 Simulation**", type="primary", help="Click to start the quantum key distribution simulation with the above parameters")

    if run_button:
        st.session_state.simulation_run = True
        st.session_state.simulation_completed = False
        st.session_state.balloons_shown = False
        st.session_state.show_detailed = False
        st.session_state.show_errors = False
        st.session_state.show_stats = False
        st.session_state.show_params = False
        st.session_state.sim_results = None
        st.session_state.last_num_bits = num_bits

    # ============================================================
    # SIMULATION LOGIC
    # ============================================================
    if st.session_state.simulation_run and not st.session_state.simulation_completed:

        with st.spinner("🔬 **Running BB84 Quantum Simulation...**"):
            progress_bar = st.progress(0)
            progress_bar.progress(25, text="Initializing quantum simulator...")

            sim = BB84Simulator()
            
            alice_bits = np.random.randint(0, 2, num_bits)
            alice_bases = np.random.randint(0, 2, num_bits)
            bob_bases = np.random.randint(0, 2, num_bits)
            
            st.session_state.alice_bits_stored = alice_bits
            st.session_state.alice_bases_stored = alice_bases
            st.session_state.bob_bases_stored = bob_bases

            progress_bar.progress(50, text="Simulating quantum transmission...")
            bob_no_eve, eve_results_no = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=False, noise_prob=noise_prob)
            bob_eve, eve_results_eve = sim.simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present=True, eve_intercept_prob=eve_prob, noise_prob=noise_prob)
            progress_bar.progress(75, text="Analyzing results and generating reports...")
            
            def compute(t_bob):
                timeline = create_transmission_timeline(alice_bits, alice_bases, bob_bases, t_bob)
                used = timeline[timeline["Used"] == True]
                
                errors = int(np.sum(used["Error"].values))
                qber = errors / len(used) if len(used) > 0 else 0.0
                sec = sim.assess_security(float(qber), float(threshold))
                status = sec['status']

                sifted_key = used["AliceBit"].astype(int).tolist()
                final_key = sim.privacy_amplification(sifted_key, qber) if status == "SECURE" else []

                return timeline, errors, qber, status, len(used), len(final_key)

            tl_no_eve, err_no, qber_no, stat_no, sift_no, key_no = compute(bob_no_eve)
            tl_eve, err_e, qber_e, stat_e, sift_e, key_e = compute(bob_eve)
            
            st.session_state.sim_results = {
                'tl_no_eve': tl_no_eve,
                'tl_eve': tl_eve,
                'err_no': err_no,
                'err_e': err_e,
                'qber_no': qber_no,
                'qber_e': qber_e,
                'stat_no': stat_no,
                'stat_e': stat_e,
                'sift_no': sift_no,
                'sift_e': sift_e,
                'key_no': key_no,
                'key_e': key_e,
                'bob_no_eve': bob_no_eve,
                'bob_eve': bob_eve,
                'eve_results_no': eve_results_no,
                'eve_results_eve': eve_results_eve
            }

            progress_bar.empty()

            st.success("✅ **Simulation completed successfully!**")
            st.session_state.simulation_completed = True

            # ============================================================
            # ============================================================
            # RESULTS SECTION
            # ============================================================
            
            # ============================================================
            # RESULTS DISPLAY
            # ============================================================
            if st.session_state.sim_results is not None:
                tl_no_eve = st.session_state.sim_results['tl_no_eve']
                tl_eve = st.session_state.sim_results['tl_eve']
                err_no = st.session_state.sim_results['err_no']
                err_e = st.session_state.sim_results['err_e']
                qber_no = st.session_state.sim_results['qber_no']
                qber_e = st.session_state.sim_results['qber_e']
                stat_no = st.session_state.sim_results['stat_no']
                stat_e = st.session_state.sim_results['stat_e']
                sift_no = st.session_state.sim_results['sift_no']
                sift_e = st.session_state.sim_results['sift_e']
                key_no = st.session_state.sim_results['key_no']
                key_e = st.session_state.sim_results['key_e']
                
                # ============================================================
                # MAIN METRICS DISPLAY
                # ============================================================
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("✅ **No Eavesdropper Scenario**")
                    st.metric("📡 Transmitted Qubits", num_bits)
                    st.metric("🔗 Sifted Bits", sift_no)
                    st.metric("❌ Errors Detected", err_no)
                    st.metric("📊 Quantum Bit Error Rate", f"{qber_no:.4f}")
                    st.metric("🔐 Final Secure Key", key_no)
                    st.metric("⚡ Key Generation Rate", f"{key_no / num_bits:.4f}" if num_bits > 0 else "0")
                    st.plotly_chart(qber_gauge(qber_no, threshold), use_container_width=True, key="gauge_no_metric")

                with c2:
                    st.subheader("🕵️ **Eavesdropper Present Scenario**")
                    st.metric("📡 Transmitted Qubits", num_bits)
                    st.metric("🔗 Sifted Bits", sift_e)
                    st.metric("❌ Errors Detected", err_e)
                    st.metric("📊 Quantum Bit Error Rate", f"{qber_e:.4f}")
                    st.metric("🔐 Final Secure Key", key_e)
                    st.metric("⚡ Key Generation Rate", f"{key_e / num_bits:.4f}" if num_bits > 0 else "0")
                    st.plotly_chart(qber_gauge(qber_e, threshold), use_container_width=True, key="gauge_e_metric")

                # ============================================================
                # DETAILED PERFORMANCE METRICS (Always shown)
                # ============================================================
                st.markdown("---")
                st.markdown("### 📊 **Detailed Performance Metrics**")
                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.markdown("**No Eve:**")
                    st.info(f"• Efficiency: {sift_no/num_bits:.1%}\n• Security: {stat_no}\n• Key Rate: {key_no/num_bits:.3f}")
                with det_col2:
                    st.markdown("**With Eve:**")
                    st.info(f"• Efficiency: {sift_e/num_bits:.1%}\n• Security: {stat_e}\n• Key Rate: {key_e/num_bits:.3f}")

                # ============================================================
                # ERROR PATTERN ANALYSIS (Always shown)
                # ============================================================
                st.markdown("---")
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

                # ============================================================
                # KEY GENERATION STATISTICS (Always shown)
                # ============================================================
                st.markdown("---")
                st.markdown("### 📈 **Key Generation Statistics**")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("**Raw Key Length**", max(key_no, key_e))
                with stat_col2:
                    st.metric("**Privacy Amplification**", f"{max(key_no, key_e)/max(sift_no, sift_e):.2f}x" if max(sift_no, sift_e) > 0 else "N/A")
                with stat_col3:
                    st.metric("**Security Level**", f"2^(-{int(np.log2(1e-6))})")
                
                # ============================================================
                # SIMULATION PARAMETERS (Always shown)
                # ============================================================
                st.markdown("---")
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
                
                # ============================================================
                # QBER ANALYSIS
                # ============================================================
                st.markdown("---")
                if sift_no > 0 or sift_e > 0:
                    st.markdown("### 📊 **QBER Analysis**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if sift_no > 0:
                        window_size = max(1, sift_no // 10)
                        local_qbers_no = []
                        for i in range(0, sift_no, window_size):
                            a_w = tl_no_eve[tl_no_eve["Used"] == True]["AliceBit"].iloc[i:i + window_size].values
                            b_w = tl_no_eve[tl_no_eve["Used"] == True]["BobResult"].iloc[i:i + window_size].values
                            if len(a_w) > 0:
                                local_qbers_no.append(float(np.mean(a_w != b_w)))
                        ax.plot(local_qbers_no, 'bo-', label=f'No Eve Local QBER (Overall: {qber_no:.3f})')

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

                    # Sifted Bits Table
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
                            st.dataframe(df_no, key="sifted_df_no")
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
                            st.dataframe(df_e, key="sifted_df_e")
                        else:
                            st.info("No sifted bits.")

                    st.plotly_chart(decision_line(qber_e, threshold, "**Attack Detection Decision Analysis**"),
                                    use_container_width=True, key="dec_line_chart")

                # ============================================================
                # TABS SECTION - Analysis & Visualization
                # ============================================================
                tab2, tab3, tab4, tab5, tab6 = st.tabs(["Timeline Analysis", "Comparative Analysis", "Quantum Visualization", "Report Generation", "Protocol Guide"])

                # Timeline Analysis Tab
                with tab2:
                    render_timeline_interactive()

                # Comparison Tab
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

                    # Download buttons
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        if key_no > 0:
                            key_no_bytes = bytes(key_no_str, 'utf-8')
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

                # Quantum Visualization Tab
                with tab4:
                    st.markdown("### 🧿 **Quantum Visualization**")

                    qv_tab1, qv_tab2, qv_tab3 = st.tabs(["Single Qubit Analysis", "Multi-Qubit Comparison", "Polarization Analysis"])

                    with qv_tab1:
                        render_single_qubit_bloch()

                    with qv_tab2:
                        render_multi_qubit_bloch()

                    with qv_tab3:
                        st.markdown("### 🧿 **Polarization Analysis**")
                        pol_col1, pol_col2 = st.columns(2)
                        
                        with pol_col1:
                            st.subheader("Rectilinear Polarization (Z-Basis)")
                            st.markdown("""
                            **Z-Basis States:**
                            - **|0⟩ (North Pole)**: Horizontal polarization, represents classical bit 0
                            - **|1⟩ (South Pole)**: Vertical polarization, represents classical bit 1
                            """)
                            try:
                                sv0 = Statevector.from_label('0')
                                sv1 = Statevector.from_label('1')
                                fig_rect = plotly_bloch_sphere([sv0, sv1])
                                st.plotly_chart(fig_rect, use_container_width=True)
                                st.markdown("**States:** |0⟩ (North Pole), |1⟩ (South Pole)")
                            except Exception as e:
                                st.error(f"❌ Failed to load rectilinear Bloch sphere: {e}")
                            
                            z_bits = [i for i, b in enumerate(st.session_state.alice_bases_stored) if b == 0]
                            z_0 = sum(1 for i in z_bits if st.session_state.alice_bits_stored[i] == 0)
                            z_1 = sum(1 for i in z_bits if st.session_state.alice_bits_stored[i] == 1)
                            st.markdown(f"**Bits in Z-Basis:** {len(z_bits)} (|0⟩: {z_0}, |1⟩: {z_1})")

                        with pol_col2:
                            st.subheader("Diagonal Polarization (X-Basis)")
                            st.markdown("""
                            **X-Basis States:**
                            - **|+⟩ (East)**: Equal superposition of |0⟩ and |1⟩ with + phase
                            - **|-⟩ (West)**: Equal superposition of |0⟩ and |1⟩ with - phase
                            """)
                            try:
                                sv_plus = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
                                sv_minus = Statevector([1/np.sqrt(2), -1/np.sqrt(2)])
                                fig_diag = plotly_bloch_sphere([sv_plus, sv_minus])
                                st.plotly_chart(fig_diag, use_container_width=True)
                                st.markdown("**States:** |+⟩ (East), |-⟩ (West)")
                            except Exception as e:
                                st.error(f"❌ Failed to load diagonal Bloch sphere: {e}")
                            
                            x_bits = [i for i, b in enumerate(st.session_state.alice_bases_stored) if b == 1]
                            x_plus = sum(1 for i in x_bits if st.session_state.alice_bits_stored[i] == 0)
                            x_minus = sum(1 for i in x_bits if st.session_state.alice_bits_stored[i] == 1)
                            st.markdown(f"**Bits in X-Basis:** {len(x_bits)} (| + ⟩: {x_plus}, | - ⟩: {x_minus})")

                # Report Generation Tab
                with tab5:
                    render_report_generation()

                # Protocol Guide Tab
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
                
                # Show balloons after all visualization
                if not st.session_state.balloons_shown:
                    st.balloons()
                    st.session_state.balloons_shown = True


if __name__ == "__main__":
    main()
