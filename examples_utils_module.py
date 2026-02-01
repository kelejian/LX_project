"""
Example Usage: Shared Utils Module
===================================

This script demonstrates how to use the shared utils module,
particularly the injury metrics (AIS calculations) and random seed management.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import shared modules
sys.path.insert(0, str(Path(__file__).parent))

from shared.utils import (
    AIS_cal_head,
    AIS_cal_chest,
    AIS_cal_neck,
    set_random_seed,
    GLOBAL_SEED,
)


def example_head_injury():
    """Example: Head injury (HIC) to AIS calculation"""
    print("=" * 60)
    print("Example 1: Head Injury Assessment (HIC → AIS)")
    print("=" * 60)
    
    # Test with different HIC values
    hic_values = [100, 400, 700, 1000, 1500, 2000]
    
    print("HIC15 Value | AIS Level | Interpretation")
    print("-" * 50)
    
    for hic in hic_values:
        ais = AIS_cal_head(hic)
        interpretation = {
            0: "No injury",
            1: "Minor",
            2: "Moderate",
            3: "Serious",
            4: "Severe",
            5: "Critical"
        }.get(ais, "Unknown")
        
        print(f"{hic:>11} | {ais:>9} | {interpretation}")
    
    print("\n✓ Head injury calculations complete")
    print()


def example_chest_injury():
    """Example: Chest injury (Dmax) to AIS calculation"""
    print("=" * 60)
    print("Example 2: Chest Injury Assessment (Dmax → AIS)")
    print("=" * 60)
    
    # Test with different Dmax values and occupant types
    dmax_values = [10, 20, 30, 40, 50, 60]
    occupant_types = [
        (1, "5th Female"),
        (2, "50th Male"),
        (3, "95th Male")
    ]
    
    for ot_code, ot_name in occupant_types:
        print(f"\nOccupant Type: {ot_name} (OT={ot_code})")
        print("Dmax (mm) | AIS Level | Interpretation")
        print("-" * 45)
        
        for dmax in dmax_values:
            ais = AIS_cal_chest(dmax, ot_code)
            interpretation = {
                0: "No injury",
                1: "Minor",
                2: "Moderate",
                3: "Serious",
                4: "Severe",
                5: "Critical"
            }.get(ais, "Unknown")
            
            print(f"{dmax:>9} | {ais:>9} | {interpretation}")
    
    print("\n✓ Chest injury calculations complete")
    print()


def example_neck_injury():
    """Example: Neck injury (Nij) to AIS calculation"""
    print("=" * 60)
    print("Example 3: Neck Injury Assessment (Nij → AIS)")
    print("=" * 60)
    
    # Test with different Nij values
    nij_values = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    print("Nij Value | AIS Level | Interpretation")
    print("-" * 45)
    
    for nij in nij_values:
        ais = AIS_cal_neck(nij)
        interpretation = {
            0: "No injury",
            1: "Minor",
            2: "Moderate",
            3: "Serious",
            4: "Severe",
            5: "Critical"
        }.get(ais, "Unknown")
        
        print(f"{nij:>9.2f} | {ais:>9} | {interpretation}")
    
    print("\n✓ Neck injury calculations complete")
    print()


def example_batch_calculations():
    """Example: Batch calculations with arrays"""
    print("=" * 60)
    print("Example 4: Batch Injury Calculations")
    print("=" * 60)
    
    # Simulate batch of injury data from multiple crash cases
    n_cases = 10
    
    hic_values = np.random.uniform(300, 1500, n_cases)
    dmax_values = np.random.uniform(20, 60, n_cases)
    nij_values = np.random.uniform(0.3, 1.5, n_cases)
    ot_values = np.random.choice([1, 2, 3], n_cases)  # Random occupant types
    
    # Calculate AIS levels for all cases at once
    head_ais = AIS_cal_head(hic_values)
    chest_ais = AIS_cal_chest(dmax_values, ot_values)
    neck_ais = AIS_cal_neck(nij_values)
    
    # Calculate MAIS (Maximum AIS)
    mais = np.maximum.reduce([head_ais, chest_ais, neck_ais])
    
    print(f"Processed {n_cases} crash cases:")
    print("\nCase | HIC  | Dmax | Nij  | OT | Head | Chest | Neck | MAIS")
    print("-" * 70)
    
    for i in range(n_cases):
        print(f"{i+1:>4} | {hic_values[i]:>4.0f} | {dmax_values[i]:>4.1f} | "
              f"{nij_values[i]:>4.2f} | {ot_values[i]:>2} | "
              f"{head_ais[i]:>4} | {chest_ais[i]:>5} | {neck_ais[i]:>4} | {mais[i]:>4}")
    
    print("\n✓ Batch calculations complete")
    print(f"  Average MAIS: {mais.mean():.2f}")
    print(f"  Max MAIS: {mais.max()}")
    print(f"  Cases with MAIS ≥ 3: {np.sum(mais >= 3)}")
    print()


def example_random_seed():
    """Example: Random seed management"""
    print("=" * 60)
    print("Example 5: Random Seed Management")
    print("=" * 60)
    
    print(f"Global seed: {GLOBAL_SEED}")
    
    # Set random seed
    set_random_seed(42)
    print("✓ Random seed set to 42")
    
    # Generate some random numbers
    random_nums_1 = np.random.randn(5)
    print(f"\nFirst generation: {random_nums_1}")
    
    # Reset seed and generate again - should be same
    set_random_seed(42)
    random_nums_2 = np.random.randn(5)
    print(f"Second generation: {random_nums_2}")
    
    # Verify reproducibility
    if np.allclose(random_nums_1, random_nums_2):
        print("✓ Random number generation is reproducible!")
    else:
        print("✗ Random numbers differ (unexpected)")
    
    print()


def example_complete_injury_assessment():
    """Example: Complete injury assessment workflow"""
    print("=" * 60)
    print("Example 6: Complete Injury Assessment Workflow")
    print("=" * 60)
    
    # Simulate a crash case
    case_id = 12345
    impact_velocity = 55.0  # km/h
    impact_angle = 0.0  # degrees
    overlap = 100.0  # %
    
    # Injury metrics from simulation/prediction
    hic = 850.0
    dmax = 45.0
    nij = 0.8
    occupant_type = 2  # 50th male
    
    print(f"Crash Case ID: {case_id}")
    print(f"Impact Conditions:")
    print(f"  - Velocity: {impact_velocity} km/h")
    print(f"  - Angle: {impact_angle}°")
    print(f"  - Overlap: {overlap}%")
    print(f"\nInjury Metrics:")
    print(f"  - HIC15: {hic}")
    print(f"  - Dmax: {dmax} mm")
    print(f"  - Nij: {nij}")
    print(f"  - Occupant Type: {occupant_type} (50th Male)")
    
    # Calculate AIS levels
    head_ais = AIS_cal_head(hic)
    chest_ais = AIS_cal_chest(dmax, occupant_type)
    neck_ais = AIS_cal_neck(nij)
    mais = max(head_ais, chest_ais, neck_ais)
    
    print(f"\nInjury Assessment (AIS):")
    print(f"  - Head: AIS {head_ais}")
    print(f"  - Chest: AIS {chest_ais}")
    print(f"  - Neck: AIS {neck_ais}")
    print(f"  - MAIS: {mais}")
    
    # Risk classification
    risk_level = {
        0: "No Risk",
        1: "Low Risk",
        2: "Moderate Risk",
        3: "High Risk",
        4: "Very High Risk",
        5: "Critical Risk"
    }.get(mais, "Unknown")
    
    print(f"\nOverall Risk: {risk_level}")
    
    # Safety recommendation
    if mais >= 3:
        print("⚠ WARNING: Serious injury risk detected!")
        print("   Recommend: Design modification or restraint system optimization")
    elif mais >= 2:
        print("⚠ CAUTION: Moderate injury risk")
        print("   Recommend: Review and consider design improvements")
    else:
        print("✓ Acceptable injury risk level")
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LX_project Shared Utils Module - Usage Examples")
    print("=" * 60 + "\n")
    
    example_head_injury()
    example_chest_injury()
    example_neck_injury()
    example_batch_calculations()
    example_random_seed()
    example_complete_injury_assessment()
    
    print("=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
