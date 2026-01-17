# Project Architecture Guide - BB84 QKD Simulator

## 📦 Directory Structure

```
bb84_2/
│
├── bb84_2.py                          ⭐ MAIN APPLICATION (1,004 lines)
│   ├── Imports (Organized)
│   ├── main() function
│   ├── Header & UI Setup
│   ├── Session State Management
│   ├── Parameter Configuration
│   ├── Simulation Execution
│   ├── Results Display
│   ├── Analysis Tabs
│   └── Entry Point
│
├── bb84_config.py                     ⚙️ CONFIGURATION (49 lines)
│   ├── Application Settings
│   ├── Default Parameters
│   ├── Visualization Settings
│   ├── Color Schemes
│   ├── Quantum Simulator Settings
│   └── Cache Configuration
│
├── bb84_simulator.py                  🔬 QUANTUM SIMULATOR (233 lines)
│   ├── BB84Simulator Class
│   ├── encode_qubit()
│   ├── simulate_transmission()
│   ├── privacy_amplification()
│   ├── assess_security()
│   ├── get_statevector_from_bit_basis()
│   └── state_label()
│
├── bb84_utils.py                      📊 DATA UTILITIES (209 lines)
│   ├── create_transmission_timeline()
│   ├── compute_metrics()
│   ├── analyze_error_patterns()
│   ├── calculate_key_rate()
│   ├── get_basis_distribution()
│   ├── get_bit_distribution()
│   └── calculate_eve_impact()
│
├── bb84_visualizations.py             📈 VISUALIZATIONS (498 lines)
│   ├── plot_pdf_style_timeline()
│   ├── plotly_bit_timeline()
│   ├── plotly_error_timeline()
│   ├── qber_gauge()
│   ├── decision_line()
│   ├── plotly_bloch_sphere()
│   └── create_pdf_report_with_graphs()
│
├── README.md                          📖 Documentation
├── QUICKSTART.md                      🚀 Quick Start
├── REORGANIZATION_SUMMARY.md          📋 This Project Info
└── jntua_logo.png                     🖼️ University Logo
```

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (bb84_2.py)                 │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 1. Configuration Section                           │     │
│  │    ↓ Gets Parameters from bb84_config              │     │
│  └────────────────────────────────────────────────────┘     │
│                        ↓                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 2. Simulation Engine                               │     │
│  │    └→ Uses BB84Simulator (bb84_simulator.py)       │     │
│  │       - Generates random bits/bases                │     │
│  │       - Simulates quantum transmission             │     │
│  │       - Handles Eve eavesdropping                  │     │
│  └────────────────────────────────────────────────────┘     │
│                        ↓                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 3. Data Processing                                 │     │
│  │    └→ Uses Utils (bb84_utils.py)                   │     │
│  │       - Creates transmission timeline              │     │
│  │       - Computes QBER metrics                      │     │
│  │       - Analyzes error patterns                    │     │
│  └────────────────────────────────────────────────────┘     │
│                        ↓                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 4. Visualization & Reporting                        │     │
│  │    └→ Uses Visualizations (bb84_visualizations.py)│     │
│  │       - PDF-style timelines                        │     │
│  │       - Interactive Plotly charts                  │     │
│  │       - Bloch sphere quantum states                │     │
│  │       - PDF reports                                │     │
│  └────────────────────────────────────────────────────┘     │
│                        ↓                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 5. Results & Analysis Tabs                          │     │
│  │    - Timeline Analysis                             │     │
│  │    - Comparative Analysis                          │     │
│  │    - Quantum Visualization                         │     │
│  │    - Report Generation                             │     │
│  │    - Protocol Guide                                │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Import Structure

```python
# Main Application (bb84_2.py)
│
├── Standard Library
│   ├── io
│   ├── hashlib
│   ├── time
│   └── datetime
│
├── Scientific Computing
│   ├── numpy
│   └── pandas
│
├── Visualization
│   ├── matplotlib
│   ├── plotly
│   └── matplotlib.patches
│
├── Quantum Computing
│   ├── qiskit.QuantumCircuit
│   ├── qiskit.Statevector
│   ├── qiskit.visualization
│   ├── qiskit_aer.AerSimulator
│   └── qiskit.transpile
│
├── Web Framework
│   └── streamlit
│
└── Local Modules
    ├── bb84_config ────→ Configuration constants
    ├── bb84_simulator ──→ Quantum simulation logic
    ├── bb84_utils ─────→ Data processing utilities
    └── bb84_visualizations → Visualization & reporting
```

## 🎯 Class & Function Hierarchy

### BB84Simulator (bb84_simulator.py)
```
BB84Simulator
├── __init__()
│   └── Initialize AerSimulator
├── encode_qubit(bit, basis)
│   └── Create quantum circuit for bit+basis
├── simulate_transmission(alice_bits, alice_bases, bob_bases, eve_present, eve_intercept_prob, noise_prob)
│   ├── Batch process qubits
│   ├── Handle Eve interception
│   ├── Apply channel noise
│   └── Return bob_results, eve_results
├── privacy_amplification(sifted_key, error_rate, target_security_level)
│   ├── Calculate Shannon entropy
│   ├── Hash sifted key (SHA-256)
│   └── Return secure key bits
├── assess_security(qber, threshold)
│   ├── Compare QBER to threshold
│   └── Return security assessment
├── get_statevector_from_bit_basis(bit, basis)
│   └── Generate quantum statevector
└── state_label(bit, basis)
    └── Return human-readable state notation
```

### Utility Functions (bb84_utils.py)
```
Data Processing Functions
├── create_transmission_timeline(alice_bits, alice_bases, bob_bases, bob_results)
│   └── Return: pd.DataFrame with full transmission details
├── compute_metrics(timeline_df, qber_threshold)
│   └── Return: dict with sifted_count, error_count, qber, etc.
├── analyze_error_patterns(timeline_df)
│   └── Return: dict with error_indices, error_count, etc.
├── calculate_key_rate(sifted_bits, final_key_length, total_qubits)
│   └── Return: dict with sifted_rate, key_rate, amplification_factor
├── get_basis_distribution(alice_bases)
│   └── Return: dict with z_basis_count, x_basis_count, percentages
├── get_bit_distribution(alice_bits)
│   └── Return: dict with zero_count, one_count, percentages
└── calculate_eve_impact(timeline_no_eve, timeline_eve)
    └── Return: dict with qber_increase, error_increase_percent, eve_detected
```

### Visualization Functions (bb84_visualizations.py)
```
Visualization Functions
├── plot_pdf_style_timeline(timeline_df, title, max_bits)
│   └── Return: matplotlib.figure.Figure (3-panel PDF-style plot)
├── plotly_bit_timeline(timeline_df, start, end, title)
│   └── Return: plotly.Figure (interactive bit comparison)
├── plotly_error_timeline(timeline_df, start, end, title)
│   └── Return: plotly.Figure (error bar chart)
├── qber_gauge(qber, threshold)
│   └── Return: plotly.Figure (gauge plot)
├── decision_line(qber, threshold, title)
│   └── Return: plotly.Figure (QBER vs threshold)
├── plotly_bloch_sphere(states)
│   └── Return: plotly.Figure (3D Bloch sphere)
├── get_statevector_from_bit_basis(bit, basis)
│   └── Return: Statevector
├── state_label(bit, basis)
│   └── Return: str (state notation)
└── create_pdf_report_with_graphs(...)
    └── Return: bytes (PDF document)
```

## 🔌 Configuration Variables (bb84_config.py)

```
CONFIG
├── Application
│   ├── APP_TITLE
│   ├── LAYOUT
│   ├── UNIVERSITY
│   └── COLLEGE
├── Defaults
│   ├── DEFAULT_QUBITS
│   ├── DEFAULT_QBER_THRESHOLD
│   ├── DEFAULT_EVE_PROB
│   ├── DEFAULT_NOISE_PROB
│   ├── DEFAULT_WINDOW_SIZE
│   ├── DEFAULT_ANIMATION_SPEED
│   ├── DEFAULT_PDF_MAX_BITS
│   └── DEFAULT_SIFTED_DISPLAY_SIZE
├── Constraints
│   ├── MIN_QUBITS
│   ├── MAX_QUBITS
│   ├── MIN_THRESHOLD
│   ├── MAX_THRESHOLD
│   ├── BATCH_SIZE
│   └── TARGET_SECURITY_LEVEL
├── Visualization
│   ├── BLOCH_SPHERE_HEIGHT
│   ├── GAUGE_HEIGHT
│   ├── TIMELINE_HEIGHT
│   ├── COLOR_GRADIENT_*
│   └── COLOR_ACCENT
├── Quantum Simulator
│   ├── SIMULATOR_METHOD
│   ├── SIMULATOR_DEVICE
│   └── SIMULATOR_SHOTS
└── Eve Attack
    └── EVE_ATTACK_TYPES
```

## 🔀 Main Application Flow (bb84_2.py)

```
main()
│
├── 1. CONFIGURATION
│   ├── Streamlit page config
│   ├── CSS styling
│   └── Load logo
│
├── 2. SESSION STATE INITIALIZATION
│   ├── Animation flags
│   ├── Simulation state
│   ├── Parameter storage
│   └── Results cache
│
├── 3. DISPLAY SECTIONS
│   ├── Header (university info)
│   ├── Information (BB84 explanation)
│   ├── Parameters (user input sliders)
│   └── Run button
│
├── 4. SIMULATION EXECUTION
│   ├── Generate random bits/bases
│   ├── Create simulator instance
│   ├── Run transmission (no Eve)
│   ├── Run transmission (with Eve)
│   ├── Compute metrics
│   └── Store results
│
├── 5. ANIMATION
│   ├── Display step-by-step process
│   ├── Show quantum states
│   └── Animate transitions
│
├── 6. RESULTS DISPLAY
│   ├── Show metrics cards
│   ├── Display QBER gauges
│   ├── Show error analysis
│   └── Show key statistics
│
├── 7. ANALYSIS TABS
│   ├── Tab 1: Timeline Analysis
│   │   ├── PDF-style plots
│   │   └── Interactive Plotly
│   ├── Tab 2: Comparative Analysis
│   │   ├── Bar charts
│   │   └── Key comparison
│   ├── Tab 3: Quantum Visualization
│   │   ├── Single qubit
│   │   ├── Multi-qubit
│   │   └── Polarization
│   ├── Tab 4: Report Generation
│   │   ├── CSV downloads
│   │   └── PDF report
│   └── Tab 5: Protocol Guide
│       └── Educational content
│
└── 8. CELEBRATION
    └── Show balloons animation
```

## ✨ Key Improvements Over Monolithic Version

| Aspect | Before | After |
|--------|--------|-------|
| **File Organization** | 1 file | 5 focused files |
| **Code Reusability** | Low | High |
| **Testing** | Difficult | Easy (per-module) |
| **Maintenance** | Hard to navigate | Clear structure |
| **Configuration** | Hardcoded | Centralized |
| **Visualization** | Mixed with logic | Separate module |
| **Scalability** | Limited | Extensible |
| **Code Clarity** | Confusing | Crystal clear |
| **File Size** | 75 KB | 52 KB main file |
| **Performance** | Good | Good + Modular |

---

**This architecture follows professional Python development best practices!**
