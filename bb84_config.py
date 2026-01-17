# Configuration file for BB84 QKD Simulator

# Application Settings
APP_TITLE = "bb84-qkd-simulator-jntu"
APP_ICON = "jntua_logo.png"
LAYOUT = "wide"

# University Info
UNIVERSITY = "Jawaharlal Nehru Technological University Anantapur"
COLLEGE = "JNTUACEA"
DEPARTMENT = "Department of Electronics and Communication Engineering"
PROJECT_NAME = "BB84 Quantum Key Distribution Simulator"
PROJECT_CODE = "AQVH FINAL"

# Simulation Parameters - Defaults
DEFAULT_QUBITS = 200
DEFAULT_QBER_THRESHOLD = 0.11
DEFAULT_EVE_PROB = 0.5
DEFAULT_NOISE_PROB = 0.01
DEFAULT_WINDOW_SIZE = 80
DEFAULT_ANIMATION_SPEED = 0.08
DEFAULT_PDF_MAX_BITS = 50
DEFAULT_SIFTED_DISPLAY_SIZE = 20

# Simulation Constraints
MIN_QUBITS = 50
MAX_QUBITS = 2000
MIN_THRESHOLD = 0.00
MAX_THRESHOLD = 0.25
BATCH_SIZE = 500  # Optimized for faster parallel processing (increased from 200)
TARGET_SECURITY_LEVEL = 1e-6
ENABLE_CACHING = True  # Cache simulation results
SIMULATION_TIMEOUT = 300  # Seconds

# Visualization Settings
BLOCH_SPHERE_HEIGHT = 600
GAUGE_HEIGHT = 320
TIMELINE_HEIGHT = 420
ERROR_TIMELINE_HEIGHT = 280

# Colors
COLOR_GRADIENT_1_START = "#667eea"
COLOR_GRADIENT_1_END = "#764ba2"
COLOR_GRADIENT_2_START = "#0f62fe"
COLOR_GRADIENT_2_END = "#0a3a5c"
COLOR_ACCENT = "#00d4ff"

# Quantum Simulator Backend
SIMULATOR_METHOD = "statevector"
SIMULATOR_DEVICE = "GPU"  # Use GPU if available, fallback to CPU
SIMULATOR_DEVICE = "CPU"
SIMULATOR_SHOTS = 1

# Eve Attack Types
EVE_ATTACK_TYPES = ["Intercept-Resend"]

# PDF Report Settings
PDF_PAPER_SIZE = (8.27, 11.69)  # A4
PDF_TITLE_FONTSIZE = 14
PDF_TEXT_FONTSIZE = 11
PDF_MONOSPACE_FONT = "monospace"

# Cache Settings
CACHE_MAX_SIZE = 5
CACHE_EXPIRY_MINUTES = 30
