# BB84 Utils Module - Data Processing and Analysis
import numpy as np
import pandas as pd
import bb84_config as config

def create_transmission_timeline(alice_bits, alice_bases, bob_bases, bob_results):
    """Create detailed timeline DataFrame for BB84 transmission (OPTIMIZED)
    
    Args:
        alice_bits: Alice's transmitted bits
        alice_bases: Alice's bases
        bob_bases: Bob's bases
        bob_results: Bob's measurement results
    
    Returns:
        pd.DataFrame: Timeline with full transmission details
    """
    alice_bits = np.asarray(alice_bits, dtype=np.int8)
    alice_bases = np.asarray(alice_bases, dtype=np.int8)
    bob_bases = np.asarray(bob_bases, dtype=np.int8)
    bob_results = np.asarray(bob_results, dtype=np.int8)
    
    # Vectorized operations for maximum speed
    matches = (alice_bases == bob_bases)
    errors = (alice_bits != bob_results) & matches
    
    n = len(alice_bits)
    timeline_data = {
        'BitIndex': np.arange(n, dtype=np.int32),
        'AliceBit': alice_bits,
        'AliceBasis': alice_bases,
        'BobBasis': bob_bases,
        'BobResult': bob_results,
        'BaseMatch': matches,
        'Used': matches,
        'Error': errors,
        'ErrorType': np.where(~matches, 'Basis Mismatch', 
                             np.where(errors, 'Transmission Error', 'None'))
    }
    
    return pd.DataFrame(timeline_data)

def compute_metrics(timeline_df, qber_threshold):
    """Compute detailed metrics from timeline (OPTIMIZED)
    
    Args:
        timeline_df: Timeline DataFrame
        qber_threshold: QBER threshold value
    
    Returns:
        dict: Computed metrics
    """
    used = timeline_df[timeline_df["Used"] == True]
    
    if len(used) == 0:
        return {
            'sifted_count': 0,
            'error_count': 0,
            'qber': 0.0,
            'accuracy': 0.0,
            'efficiency': 0.0,
            'correct_count': 0,
            'is_secure': True
        }
    
    error_count = int(np.sum(used["Error"].values))
    qber = error_count / len(used) if len(used) > 0 else 0.0
    correct_count = len(used) - error_count
    
    return {
        'sifted_count': len(used),
        'error_count': error_count,
        'correct_count': correct_count,
        'qber': float(qber),
        'accuracy': correct_count / len(used) if len(used) > 0 else 0.0,
        'efficiency': len(used) / len(timeline_df) if len(timeline_df) > 0 else 0.0,
        'is_secure': qber <= qber_threshold
    }

def analyze_error_patterns(timeline_df):
    """Analyze error distribution patterns (OPTIMIZED)
    
    Args:
        timeline_df: Timeline DataFrame
    
    Returns:
        dict: Error analysis statistics
    """
    used = timeline_df[timeline_df["Used"] == True]
    
    if len(used) == 0:
        return {
            'error_indices': [],
            'error_count': 0,
            'consecutive_errors': 0,
            'error_percentage': 0.0
        }
    
    errors = used[used["Error"] == True]
    error_indices = errors["BitIndex"].tolist()
    
    # Calculate consecutive error runs with vectorization
    consecutive_errors = 0
    if len(error_indices) > 0:
        error_array = np.array(error_indices)
        diffs = np.diff(error_array)
        # Find where consecutive indices exist (diff == 1)
        consecutive_runs = np.split(error_array, np.where(diffs != 1)[0] + 1)
        consecutive_errors = max([len(run) for run in consecutive_runs]) if consecutive_runs else 0
    
    return {
        'error_indices': error_indices[:20],
        'error_count': len(errors),
        'consecutive_errors': consecutive_errors,
        'error_percentage': (len(errors) / len(used) * 100) if len(used) > 0 else 0.0
    }

def calculate_key_rate(sifted_bits, final_key_length, total_qubits):
    """Calculate quantum key rate metrics
    
    Args:
        sifted_bits: Number of sifted bits
        final_key_length: Length of final secure key
        total_qubits: Total qubits transmitted
    
    Returns:
        dict: Key rate metrics
    """
    return {
        'sifted_rate': sifted_bits / total_qubits if total_qubits > 0 else 0,
        'key_rate': final_key_length / total_qubits if total_qubits > 0 else 0,
        'amplification_factor': final_key_length / sifted_bits if sifted_bits > 0 else 0,
        'security_level': -np.log2(config.TARGET_SECURITY_LEVEL)
    }

def get_basis_distribution(alice_bases):
    """Analyze basis distribution (OPTIMIZED)
    
    Args:
        alice_bases: Array of basis choices
    
    Returns:
        dict: Basis statistics
    """
    alice_bases = np.asarray(alice_bases, dtype=np.int8)
    z_basis = np.sum(alice_bases == 0)
    x_basis = np.sum(alice_bases == 1)
    total = len(alice_bases)
    
    return {
        'z_basis_count': int(z_basis),
        'x_basis_count': int(x_basis),
        'z_basis_percent': float(z_basis / total * 100) if total > 0 else 0.0,
        'x_basis_percent': float(x_basis / total * 100) if total > 0 else 0.0,
        'total': int(total)
    }

def get_bit_distribution(alice_bits):
    """Analyze bit distribution (OPTIMIZED)
    
    Args:
        alice_bits: Array of bit values
    
    Returns:
        dict: Bit statistics
    """
    alice_bits = np.asarray(alice_bits, dtype=np.int8)
    zeros = np.sum(alice_bits == 0)
    ones = np.sum(alice_bits == 1)
    total = len(alice_bits)
    
    return {
        'zero_count': int(zeros),
        'one_count': int(ones),
        'zero_percent': float(zeros / total * 100) if total > 0 else 0.0,
        'one_percent': float(ones / total * 100) if total > 0 else 0.0,
        'total': int(total)
    }

def calculate_eve_impact(timeline_no_eve, timeline_eve):
    """Calculate impact of Eve's eavesdropping (OPTIMIZED)
    
    Args:
        timeline_no_eve: Timeline without Eve
        timeline_eve: Timeline with Eve
    
    Returns:
        dict: Eve impact analysis
    """
    used_no_eve = np.sum(timeline_no_eve["Used"].values)
    used_eve = np.sum(timeline_eve["Used"].values)
    
    errors_no_eve = np.sum(timeline_no_eve["Error"].values)
    errors_eve = np.sum(timeline_eve["Error"].values)
    
    qber_no = errors_no_eve / used_no_eve if used_no_eve > 0 else 0.0
    qber_eve = errors_eve / used_eve if used_eve > 0 else 0.0
    
    error_increase = errors_eve - errors_no_eve
    error_increase_percent = (error_increase / max(1, errors_no_eve) * 100) if errors_no_eve > 0 else (100.0 if errors_eve > 0 else 0.0)
    
    return {
        'qber_increase': float(qber_eve - qber_no),
        'error_increase_percent': float(error_increase_percent),
        'qber_no_eve': float(qber_no),
        'qber_with_eve': float(qber_eve),
        'eve_detected': qber_eve > config.DEFAULT_QBER_THRESHOLD
    }
