# BB84 Quantum Key Distribution Simulator

## 📚 Overview

This is an advanced Streamlit-based simulator for the BB84 Quantum Key Distribution protocol, developed as a final project for AQVH. The simulator demonstrates the principles of quantum cryptography, allowing users to explore secure key exchange between Alice and Bob, with optional eavesdropping by Eve.

## 🎯 Features

### Core Simulation
- **Quantum Bit Transmission**: Simulate sending qubits with random bits and bases
- **Eavesdropping Scenarios**: Include optional Eve intercept-resend attacks
- **Channel Noise**: Add realistic noise to quantum channels
- **Security Analysis**: Automatic QBER calculation and security assessment

### Visualizations
- **Interactive Bloch Spheres**: 3D visualization of quantum states (single and multi-qubit)
- **Timeline Analysis**: Detailed transmission timelines with error tracking
- **QBER Analysis**: Local error rate analysis across bit sequences
- **Comparative Charts**: Side-by-side metrics for scenarios with/without Eve

### Advanced Features
- **Process Animation**: Step-by-step BB84 protocol animation
- **Report Generation**: PDF reports with graphs and analysis
- **Key Distribution**: Final secure key extraction and display
- **Privacy Amplification**: Hashing for key security enhancement

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `jntua_logo.png` is in the same directory as the app

### Requirements
- streamlit
- qiskit
- qiskit-aer
- plotly
- numpy
- pandas
- matplotlib

## 🎮 Usage

### Running the App
```bash
streamlit run bb84_1app.py
```

### Interface Guide

#### Sidebar Controls
- **Qubits to transmit**: Number of quantum bits (50-2000)
- **QBER Threshold**: Security threshold for error rates
- **Eve Interception Probability**: Chance of eavesdropping
- **Channel Noise Probability**: Transmission noise level
- **Timeline Parameters**: Visualization settings

#### Main Interface
1. **Adjust parameters** in the sidebar
2. **Click "Start Simulation"** to run the BB84 protocol
3. **Explore results** in the main area:
   - Performance metrics
   - Process animation
   - Detailed analysis tabs

#### Analysis Tabs
- **Timeline Analysis**: Transmission visualization
- **Comparative Analysis**: Metrics comparison
- **Quantum Visualization**: Bloch sphere analysis
- **Report Generation**: PDF export
- **Protocol Guide**: BB84 explanation

## 🔧 Configuration

### Parameters
- **Default qubits**: 50 (for fast execution)
- **Security threshold**: 0.11 (standard BB84 threshold)
- **Animation speed**: 1.0 seconds per step

### Customization
- Modify `num_bits` default in the code for different scales
- Adjust visualization limits in tab functions
- Change color schemes in plotting functions

## 📊 Technical Details

### BB84 Protocol Implementation
1. **Key Generation**: Random bits and bases
2. **Qubit Preparation**: |0⟩, |1⟩, |+⟩, |-⟩ states
3. **Quantum Transmission**: Simulated with Qiskit AerSimulator
4. **Measurement**: Bob's basis selection and measurement
5. **Post-Processing**: Basis reconciliation, error estimation, privacy amplification

### Security Analysis
- Quantum Bit Error Rate (QBER) calculation
- Eavesdropping detection threshold
- Secure key length computation using Devetak bound

## 🌐 Deployment

### Local Deployment
```bash
streamlit run bb84_1app.py --server.port 8501
```

### Cloud Deployment
- **Streamlit Cloud**: Upload files and deploy directly
- **Heroku**: Use the provided requirements.txt
- **Docker**: Create container with Python environment

### Production Considerations
- Reduce default qubit count for faster loading
- Implement caching for repeated simulations
- Add user authentication for multi-user scenarios

## 📈 Performance

### Recommended Settings
- **Fast**: 50 qubits, default parameters
- **Standard**: 200 qubits, moderate noise
- **Advanced**: 500+ qubits, full analysis

### System Requirements
- **RAM**: 2GB minimum, 4GB recommended
- **CPU**: Multi-core processor for parallel simulations
- **Storage**: 100MB for app and dependencies

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all packages are installed
2. **Logo Not Showing**: Check `jntua_logo.png` path
3. **Slow Performance**: Reduce qubit count or use faster hardware
4. **Visualization Errors**: Check browser compatibility

### Debug Mode
```bash
streamlit run bb84_1app.py --logger.level=debug
```

## 📚 Educational Value

This simulator serves as an excellent teaching tool for:
- Quantum cryptography concepts
- BB84 protocol mechanics
- Quantum state visualization
- Information security principles
- Scientific computing with Python

## 🤝 Contributing

For improvements or bug fixes:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License

This project is developed for educational purposes at Jawaharlal Nehru Technological University Anantapur.

## 🙏 Acknowledgments

- Charles Bennett and Gilles Brassard for inventing BB84
- Qiskit community for quantum computing tools
- Streamlit team for the web app framework

---

**Developed by**:  Team silicon  
**Institution**: JNTUACEA, Department of ECE  
**Date**: January 2026</content>
<parameter name="filePath">/home/keerthan/Desktop/README.md
