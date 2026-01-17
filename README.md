# BB84 Quantum Key Distribution Simulator

A comprehensive Streamlit application for simulating the BB84 quantum key distribution protocol with advanced features including attack detection, polarization analysis, and performance optimization.

## Project Files

### Core Modules
- **bb84_2.py** - Main Streamlit application with complete UI
- **bb84_simulator.py** - Quantum simulator core (Qiskit-based)
- **bb84_config.py** - Configuration and constants
- **bb84_utils.py** - Data processing and analysis utilities
- **bb84_visualizations.py** - Plotting and visualization functions

### Assets
- **jntua_logo.png** - JNTUACEA university logo

## Features

### 🔒 Core BB84 Protocol
- Alice generates random bits and bases
- Quantum transmission simulation
- Bob's measurement with basis selection
- Sifting process (matching bases only)
- Privacy amplification (SHA-256 hashing)

### 🕵️ Attack Detection
- **Parallel QBER Analysis**: Side-by-side comparison (Without Eve | With Eve)
- **Eve Impact Analysis**: Eavesdropping detection metrics
- **Attack Detection Summary**: Color-coded security status
- **QBER Threshold**: Configurable security threshold (default: 11%)

### 📊 Quantum Visualization
- **Bloch Sphere Visualization**: Interactive 3D quantum state display
- **Polarization Analysis**:
  - Rectilinear (Z-basis): |0⟩, |1⟩ states
  - Diagonal (X-basis): |+⟩, |-⟩ states
- **Single/Multi-qubit Views**: Flexible analysis options

### 📄 Report Generation
- **CSV Export**: Transmission timelines for both scenarios
- **PDF Reports**: 4-page comprehensive analysis with:
  - Summary and conclusions
  - Transmission timelines
  - Comparison charts
  - Project details

### ⚡ Performance Optimizations
- Batch processing (200 qubits/batch)
- Vectorized NumPy operations
- Cached simulator initialization
- Memory-efficient data types (int8)
- 30-50% faster execution

## Installation

### Requirements
```bash
pip install streamlit qiskit qiskit-aer numpy pandas matplotlib plotly
```

### Setup
```bash
cd /home/keerthan/Desktop/bb84_2
source bb84env/bin/activate  # If using virtual environment
streamlit run bb84_2.py
```

## Usage

1. **Configure Parameters**:
   - Number of qubits (50-2000)
   - Eve's interception probability (0-100%)
   - Channel noise (0-5%)
   - QBER threshold (0-25%)

2. **Run Simulation**:
   - Click "RUN SIMULATION"
   - Monitor progress bar
   - Results display automatically

3. **Analyze Results**:
   - View metrics dashboard
   - Check transmission timelines
   - Study QBER attack detection
   - Examine Bloch sphere visualizations
   - Review error patterns

4. **Download Reports**:
   - CSV files for No Eve scenario
   - CSV files for With Eve scenario
   - Comprehensive PDF report (4 pages)

## Performance

### Expected Execution Times
| Qubits | Time | Notes |
|--------|------|-------|
| 50-100 | <1s | Instant |
| 200 | 1-2s | Default (very fast) |
| 500 | 3-5s | Good performance |
| 1000 | 5-10s | Still responsive |
| 2000 | 15-20s | Larger simulations |

### Optimization Features
- ✅ Batch processing (3-4x faster)
- ✅ Vectorized operations (8-40x faster)
- ✅ Error analysis (10-100x faster)
- ✅ Memory reduction (8x smaller)
- ✅ Cached simulator (2x on reuse)

## Configuration

Edit `bb84_config.py` to customize:
```python
DEFAULT_QUBITS = 200              # Default qubit count
DEFAULT_QBER_THRESHOLD = 0.11     # 11% threshold
BATCH_SIZE = 200                  # Qubits per batch
DEFAULT_EVE_PROB = 0.5            # Eve probability
DEFAULT_NOISE_PROB = 0.01         # Channel noise
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         bb84_2.py (Streamlit UI)            │
├─────────────────────────────────────────────┤
│  ┌────────────────────────────────────────┐ │
│  │  bb84_simulator.py (Qiskit Core)       │ │
│  │  - Quantum circuit generation          │ │
│  │  - Measurement simulation              │ │
│  │  - Privacy amplification               │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │  bb84_utils.py (Data Processing)       │ │
│  │  - Timeline generation                 │ │
│  │  - Metric computation                  │ │
│  │  - Error analysis                      │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │  bb84_visualizations.py (Plotting)     │ │
│  │  - Bloch spheres                       │ │
│  │  - QBER gauges                         │ │
│  │  - PDF report generation               │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │  bb84_config.py (Configuration)        │ │
│  │  - Constants and defaults               │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Security Model

### BB84 Protocol Steps
1. **Preparation**: Alice generates random bits and bases
2. **Transmission**: Quantum states sent to Bob
3. **Measurement**: Bob measures with random bases
4. **Sifting**: Keep matching basis results (~50%)
5. **Error Detection**: Calculate QBER
6. **Privacy Amplification**: Hash for security

### Security Threshold
- **QBER < 11%**: Channel secure (no eavesdropping)
- **QBER > 11%**: Abort key exchange (Eve detected)

## Visualization Examples

### QBER Attack Detection
- **Without Eve**: Low QBER (~1%), GREEN indicator
- **With Eve**: High QBER (~13%), RED indicator
- **Comparison Chart**: Side-by-side bar chart
- **Summary Table**: All metrics at a glance

### Polarization Analysis
- **Rectilinear (Z-basis)**: |0⟩ North, |1⟩ South
- **Diagonal (X-basis)**: |+⟩ East, |-⟩ West
- **Interactive 3D Bloch Sphere**: Rotate and zoom
- **State Statistics**: Bit counts and distributions

## Troubleshooting

### Qiskit Import Error
```bash
pip install qiskit qiskit-aer --upgrade
```

### Slow Performance
- Reduce qubit count for testing
- Increase BATCH_SIZE in bb84_config.py
- Use CPU simulator (already configured)

### Missing Logo
Logo displays as placeholder if jntua_logo.png not found - no functional impact

## References

- Bennett & Brassard (1984) - Original BB84 Protocol
- Shor & Preskill (2000) - Security Proof
- Qiskit Documentation: https://qiskit.org/

## Author

JNTUACEA Electronics & Communication Engineering
Department of Electronics and Communication Engineering

## License

Educational/Research Use

## Version

1.0 - Complete with PDF Reports and Performance Optimization
