# BB84 Simulator Core Module
import numpy as np
import hashlib
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import bb84_config as config

class BB84Simulator:
    """Optimized BB84 Quantum Key Distribution Simulator using Qiskit"""
    
    def __init__(self):
        """Initialize the simulator with optimal settings"""
        try:
            self.simulator = AerSimulator(
                method=config.SIMULATOR_METHOD,
                device=config.SIMULATOR_DEVICE
            )
        except:
            self.simulator = AerSimulator(
                method=config.SIMULATOR_METHOD,
                device="CPU"
            )
    
    @staticmethod
    def encode_qubit(bit, basis):
        """Encode a classical bit into a quantum state
        
        Args:
            bit: Classical bit (0 or 1)
            basis: Basis choice (0=Z, 1=X)
        
        Returns:
            QuantumCircuit: Encoded quantum circuit
        """
        qc = QuantumCircuit(1, 1)
        if int(bit) == 1:
            qc.x(0)
        if int(basis) == 1:
            qc.h(0)
        return qc
    
    def simulate_transmission(self, alice_bits, alice_bases, bob_bases, 
                            eve_present=False, eve_intercept_prob=0.5, 
                            noise_prob=0.0):
        """Simulate quantum transmission of qubits (HIGHLY OPTIMIZED)
        
        Args:
            alice_bits: Alice's random bits
            alice_bases: Alice's basis choices
            bob_bases: Bob's basis choices
            eve_present: Whether Eve is eavesdropping
            eve_intercept_prob: Probability Eve intercepts each qubit
            noise_prob: Channel noise probability
        
        Returns:
            tuple: (bob_results, eve_results)
        """
        # Vectorize input conversion with efficient data types
        alice_bits = np.asarray([int(b) for b in list(alice_bits)], dtype=np.int8)
        alice_bases = np.asarray([int(b) for b in list(alice_bases)], dtype=np.int8)
        bob_bases = np.asarray([int(b) for b in list(bob_bases)], dtype=np.int8)
        
        n = len(alice_bits)
        bob_results = np.zeros(n, dtype=np.int8)
        eve_results = np.zeros(n, dtype=np.int8) if eve_present else None
        eve_bases = None
        eve_intercepts = None
        
        if eve_present:
            eve_bases = np.random.randint(0, 2, n, dtype=np.int8)
            # Pre-generate all intercept decisions for vectorization
            eve_intercepts = np.random.random(n) < eve_intercept_prob
        
        # Optimized batch processing for maximum speed
        batch_size = config.BATCH_SIZE
        
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = np.arange(batch_start, batch_end)
            
            circuits = []
            
            for i in batch_indices:
                qc = QuantumCircuit(1, 1)
                
                # Encode Alice's bit into circuit
                if alice_bits[i] == 1:
                    qc.x(0)
                
                # Handle Eve interception if present
                if eve_present and eve_intercepts[i]:
                    if eve_bases[i] == 1:
                        qc.h(0)
                    qc.measure(0, 0)
                    job_eve = self.simulator.run(transpile(qc, self.simulator), shots=1)
                    res_eve = job_eve.result().get_counts()
                    eve_results[i] = int(list(res_eve.keys())[0])
                    
                    # Resend with Eve's measurement result
                    qc = QuantumCircuit(1, 1)
                    if eve_results[i] == 1:
                        qc.x(0)
                    if eve_bases[i] == 1:
                        qc.h(0)
                else:
                    # Apply Alice's basis if not intercepted
                    if alice_bases[i] == 1:
                        qc.h(0)
                
                # Apply Bob's measurement basis
                if bob_bases[i] == 1:
                    qc.h(0)
                qc.measure(0, 0)
                circuits.append(qc)
            
            # Run batch with optimized transpilation (level 3 for maximum optimization)
            if circuits:
                transpiled = transpile(circuits, self.simulator, optimization_level=3)
                job = self.simulator.run(transpiled, shots=1)
                results = job.result()
                
                for idx, i in enumerate(batch_indices):
                    counts = results.get_counts(idx)
                    bob_results[i] = int(list(counts.keys())[0])
                    
                    # Apply channel noise efficiently
                    if noise_prob > 0 and np.random.random() < noise_prob:
                        bob_results[i] = 1 - bob_results[i]
        
        return bob_results.tolist(), eve_results.tolist() if eve_present else None
    
    @staticmethod
    def privacy_amplification(sifted_key, error_rate, 
                            target_security_level=None):
        """Apply privacy amplification via hashing (IMPROVED)
        
        Args:
            sifted_key: The sifted key bits
            error_rate: Quantum bit error rate
            target_security_level: Target security parameter
        
        Returns:
            list: Amplified secure key bits
        """
        if target_security_level is None:
            target_security_level = config.TARGET_SECURITY_LEVEL
            
        sifted_key = np.array([int(b) for b in list(sifted_key)], dtype=np.int8)
        n = len(sifted_key)
        if n == 0:
            return []

        # Improved Shannon entropy calculation
        e = float(np.clip(error_rate, 0.0, 1.0))
        if e <= 0.0 or e >= 1.0:
            h_eve = 0.0
        else:
            # Use more accurate entropy formula
            h_eve = -e * np.log2(e) - (1 - e) * np.log2(1 - e)

        # Compute secure key length with safety margin
        secure_length = n * (1 - h_eve) - 2 * np.log2(1 / float(target_security_level))
        secure_length = max(0, int(secure_length))

        if secure_length == 0:
            return []

        # Use SHA-256 for better hash properties
        key_str = ''.join(str(int(b)) for b in sifted_key)
        digest = hashlib.sha256(key_str.encode()).hexdigest()
        binary_hash = bin(int(digest, 16))[2:].zfill(256)
        
        # Extract secure bits with additional hashing if needed
        final_bits = [int(b) for b in binary_hash[:secure_length]]
        
        # If need more bits, use SHA-512 as secondary source
        if secure_length > 256:
            digest2 = hashlib.sha512(key_str.encode()).hexdigest()
            binary_hash2 = bin(int(digest2, 16))[2:].zfill(512)
            final_bits.extend([int(b) for b in binary_hash2[:secure_length - 256]])

        return final_bits[:secure_length]
    
    @staticmethod
    def assess_security(qber, threshold=None):
        """Assess security based on QBER
        
        Args:
            qber: Quantum bit error rate
            threshold: QBER threshold for security
        
        Returns:
            dict: Security assessment
        """
        if threshold is None:
            threshold = config.DEFAULT_QBER_THRESHOLD
            
        if qber <= threshold:
            return {
                'status': 'SECURE',
                'message': f'QBER ({qber:.3f}) below threshold. Key exchange successful.',
                'action': 'PROCEED_WITH_KEY',
                'color': 'green'
            }

        return {
            'status': 'INSECURE',
            'message': f'QBER ({qber:.3f}) exceeds threshold ({threshold}). Eavesdropping suspected.',
            'action': 'ABORT_AND_RETRY',
            'color': 'red'
        }
    
    @staticmethod
    def get_statevector_from_bit_basis(bit, basis):
        """Get Statevector for given bit and basis
        
        Args:
            bit: Classical bit (0 or 1)
            basis: Basis choice (0=Z, 1=X)
        
        Returns:
            Statevector: Quantum state vector
        """
        qc = QuantumCircuit(1)
        if int(bit) == 1:
            qc.x(0)
        if int(basis) == 1:
            qc.h(0)
        return Statevector.from_instruction(qc)
    
    @staticmethod
    def state_label(bit, basis):
        """Get human-readable label for quantum state
        
        Args:
            bit: Classical bit
            basis: Basis choice
        
        Returns:
            str: State label with notation
        """
        if int(basis) == 0:
            return "|0⟩ (Z)" if int(bit) == 0 else "|1⟩ (Z)"
        return "|+⟩ (X)" if int(bit) == 0 else "|−⟩ (X)"
