# BB84 Simulator - Quick Start Guide

## 📂 Project Structure

```
bb84_2/
├── bb84_2.py                    # Main Streamlit application
├── bb84_simulator.py            # Quantum simulator (Qiskit)
├── bb84_config.py               # Configuration & constants
├── bb84_utils.py                # Data processing utilities
├── bb84_visualizations.py       # Plotting & visualization
├── jntua_logo.png               # University logo
├── README.md                     # Full documentation
└── QUICKSTART.md                # This file
```

## �� Getting Started (5 Minutes)

### Step 1: Navigate to Project
```bash
cd /home/keerthan/Desktop/bb84_2
```

### Step 2: Activate Environment (if needed)
```bash
source /path/to/bb84env/bin/activate
```

### Step 3: Run the Application
```bash
streamlit run bb84_2.py
```

### Step 4: Open in Browser
Browser will automatically open to:
```
http://localhost:8501
```

## ⚙️ Configuration

### Default Parameters
- **Qubits**: 200 (fast, good for testing)
- **Eve Probability**: 50% (moderate attack)
- **Channel Noise**: 1% (realistic)
- **QBER Threshold**: 11% (BB84 standard)

### Adjust Before Running
Edit `bb84_config.py`:
```python
DEFAULT_QUBITS = 200          # Change qubit count
BATCH_SIZE = 200              # Larger = faster for big sims
DEFAULT_EVE_PROB = 0.5        # Eve probability (0-1)
DEFAULT_NOISE_PROB = 0.01     # Channel noise (0-0.05)
```

## 📊 Features Overview

### Metrics Dashboard
- Total transmitted qubits
- Sifted bits (matching bases)
- Final key length
- QBER (Quantum Bit Error Rate)
- Accuracy & efficiency

### QBER Attack Detection (PARALLEL VIEW)
- **Without Eve**: Shows clean channel (GREEN)
- **With Eve**: Shows attack scenario (RED)
- Color-coded security status
- Attack detection summary table
- Eve impact metrics

### Transmission Timeline
- Bit-by-bit transmission view
- Error detection visualization
- Basis matching indicator
- Toggle between scenarios

### Bloch Sphere Visualization
- **Single Qubit**: Pick any qubit to analyze
- **Multi-Qubit**: View 5 qubits at once
- **Polarization Analysis**: See Z-basis and X-basis states
  - Rectilinear: |0⟩, |1⟩
  - Diagonal: |+⟩, |-⟩

### Reports
- **CSV Files**: Download transmission data
- **PDF Report**: 4-page comprehensive analysis
  - Summary page with conclusions
  - Timeline visualizations
  - Comparison charts
  - Project details

## ⏱️ Expected Runtime

| Qubits | Time | Best For |
|--------|------|----------|
| 50 | <1s | Testing |
| 200 | 1-2s | Default |
| 500 | 3-5s | Analysis |
| 1000 | 5-10s | Research |
| 2000 | 15-20s | Deep study |

## 🔍 Interpreting Results

### QBER Values
- **0-2%**: Clean channel (expected without Eve)
- **2-11%**: Channel noise (investigate)
- **>11%**: Eve detected! (abort key exchange)

### With Eve Attack
- **Low QBER (<11%)**: Eve evaded detection (rare)
- **High QBER (>11%)**: Attack successful (Eve caught!)
- **QBER Increase**: Shows Eve's impact

### Security Status
- **✅ SECURE**: No attack detected, proceed
- **⚠️ ANOMALY**: High noise, investigate
- **❌ ATTACKED**: Eve detected, abort!

## 💡 Usage Examples

### Example 1: Quick Test
1. Keep default settings (200 qubits)
2. Click "RUN SIMULATION"
3. Wait ~2 seconds
4. Review metrics in dashboard

### Example 2: Study Eve's Impact
1. Set qubits to 500
2. Set Eve probability to 100%
3. Set noise to 0%
4. Run simulation
5. Compare QBER values

### Example 3: Generate Report
1. Run simulation
2. Go to "Reports" tab
3. Download CSV (data analysis)
4. Download PDF (presentation)

## 🔧 Troubleshooting

### Application Won't Start
```bash
pip install streamlit --upgrade
streamlit run bb84_2.py
```

### Qiskit Errors
```bash
pip install qiskit qiskit-aer --upgrade
```

### Slow Simulation
- Reduce qubit count (try 100)
- Increase BATCH_SIZE in config
- Check system resources

### Missing Logo
- Application still works
- Logo shows placeholder if not found
- Not critical for functionality

## 📚 Learning Resources

### In Application
- "Protocol Info" tab: Full BB84 explanation
- Expandable sections: Detailed information
- Status cards: Real-time feedback

### External Resources
- BB84 Paper (1984): Bennett & Brassard
- Qiskit Docs: https://qiskit.org/
- Quantum Cryptography Textbooks

## 🎯 Quick Tips

✅ **Do This**
- Start with 200 qubits
- Use default settings first
- Review all tabs
- Try different Eve probabilities
- Download PDF for references

❌ **Avoid This**
- Don't set qubits too high (>2000) initially
- Don't ignore QBER values
- Don't close browser during simulation
- Don't run multiple instances

## 📝 File Descriptions

| File | Purpose | Key Functions |
|------|---------|----------------|
| `bb84_2.py` | Main UI | render_header(), main() |
| `bb84_simulator.py` | Quantum core | simulate_transmission() |
| `bb84_config.py` | Settings | DEFAULT_* constants |
| `bb84_utils.py` | Data processing | compute_metrics() |
| `bb84_visualizations.py` | Plotting | qber_gauge(), plotly_* |

## 🚀 Next Steps

1. **Run first simulation** (2 min)
2. **Explore all tabs** (5 min)
3. **Try different parameters** (10 min)
4. **Generate reports** (5 min)
5. **Read full README** (10 min)

## 💬 Questions?

Refer to:
1. **README.md** - Full documentation
2. **In-app help** - Expandable sections
3. **Comments in code** - Code documentation

---

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: January 17, 2026
