"""
Main Execution Script for Graph ML Project
Combines topological GNN, interpretable GNAN, equivariant GNN, and statistical analysis
Based on latest research from JMLR 2024, NeurIPS 2024, Nature Communications, ICML 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

print("=" * 80)
print(" " * 20 + "GRAPH ML & STATISTICS PROJECT")
print(" " * 15 + "Novel Implementations from Recent Research")
print("=" * 80)
print("\nBased on recent high-impact publications:")
print("  • Line Graph Vietoris-Rips Persistence Diagram (JMLR 2024)")
print("  • Graph Neural Additive Networks (NeurIPS 2024)")
print("  • E(n) Equivariant Graph Neural Networks (Nature Comm. & ICML 2024)")
print("  • Graph Transformers & Geometric Deep Learning (ICML/ICLR 2024)")
print("=" * 80)

# Import all modules
print("\nImporting modules...")
sys.path.insert(0, 'src')

try:
    from topological_gnn import (
        PersistenceDiagramExtractor,
        TopologicalGNN,
        create_sample_graphs,
        visualize_persistence_diagram,
        visualize_barcode
    )
    print("✓ Topological GNN module loaded")
except Exception as e:
    print(f"✗ Error loading topological_gnn: {e}")

try:
    from interpretable_gnan import (
        GraphNeuralAdditiveNetwork,
        demonstrate_gnan
    )
    print("✓ Interpretable GNAN module loaded")
except Exception as e:
    print(f"✗ Error loading interpretable_gnan: {e}")

try:
    from equivariant_gnn import (
        EquivariantGNN,
        demonstrate_equivariant_gnn
    )
    print("✓ Equivariant GNN module loaded")
except Exception as e:
    print(f"✗ Error loading equivariant_gnn: {e}")

try:
    from statistical_analysis import (
        GraphStatistics,
        demonstrate_statistical_analysis
    )
    print("✓ Statistical Analysis module loaded")
except Exception as e:
    print(f"✗ Error loading statistical_analysis: {e}")

try:
    from visualizations import (
        demonstrate_visualizations
    )
    print("✓ Visualization module loaded")
except Exception as e:
    print(f"✗ Error loading visualizations: {e}")

try:
    from validation import (
        run_comprehensive_validation
    )
    print("✓ Validation module loaded")
except Exception as e:
    print(f"✗ Error loading validation: {e}")


def main():
    """Execute all components and generate outputs"""

    print("\n" + "=" * 80)
    print("EXECUTION PLAN")
    print("=" * 80)
    print("1. Topological Graph Neural Networks (JMLR 2024)")
    print("2. Graph Neural Additive Networks - Interpretable GNN (NeurIPS 2024)")
    print("3. E(n) Equivariant GNN (Nature Communications & ICML 2024)")
    print("4. Statistical Analysis of Graphs")
    print("5. Comprehensive Visualizations & Architecture Diagrams")
    print("6. Validation & Testing")
    print("=" * 80)

    results = {}

    # ========================================================================
    # 1. TOPOLOGICAL GNN
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 1: TOPOLOGICAL GRAPH NEURAL NETWORKS")
    print("▓" * 80)

    try:
        # Create sample graphs
        graphs = create_sample_graphs()
        extractor = PersistenceDiagramExtractor(max_dim=2)

        for name, G in graphs[:3]:
            print(f"\n{'─' * 70}")
            print(f"Processing: {name}")
            print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            # Compute persistence
            dgms = extractor.compute_node_persistence(G)

            if dgms is not None:
                features = extractor.extract_persistence_statistics(dgms)
                print(f"✓ Extracted {len(features)} topological features")

                # Visualize
                name_clean = name.replace(" ", "_")
                visualize_persistence_diagram(dgms,
                    f'outputs/topo_persistence_diagram_{name_clean}.png')
                visualize_barcode(dgms,
                    f'outputs/topo_barcode_{name_clean}.png')

                print(f"✓ Generated persistence visualizations")

        results['topological_gnn'] = 'SUCCESS'

    except Exception as e:
        print(f"✗ Error in Topological GNN: {e}")
        results['topological_gnn'] = f'FAILED: {e}'

    # ========================================================================
    # 2. INTERPRETABLE GNAN
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 2: GRAPH NEURAL ADDITIVE NETWORKS (Interpretable)")
    print("▓" * 80)

    try:
        model, importance = demonstrate_gnan()
        results['gnan'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Error in GNAN: {e}")
        results['gnan'] = f'FAILED: {e}'

    # ========================================================================
    # 3. EQUIVARIANT GNN
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 3: E(n) EQUIVARIANT GRAPH NEURAL NETWORKS")
    print("▓" * 80)

    try:
        model, eq_results = demonstrate_equivariant_gnn()
        results['equivariant_gnn'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Error in Equivariant GNN: {e}")
        results['equivariant_gnn'] = f'FAILED: {e}'

    # ========================================================================
    # 4. STATISTICAL ANALYSIS
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 4: ADVANCED STATISTICAL ANALYSIS")
    print("▓" * 80)

    try:
        demonstrate_statistical_analysis()
        results['statistical_analysis'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Error in Statistical Analysis: {e}")
        results['statistical_analysis'] = f'FAILED: {e}'

    # ========================================================================
    # 5. VISUALIZATIONS
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 5: COMPREHENSIVE VISUALIZATIONS")
    print("▓" * 80)

    try:
        demonstrate_visualizations()
        results['visualizations'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Error in Visualizations: {e}")
        results['visualizations'] = f'FAILED: {e}'

    # ========================================================================
    # 6. VALIDATION
    # ========================================================================
    print("\n" + "▓" * 80)
    print("MODULE 6: VALIDATION & TESTING")
    print("▓" * 80)

    try:
        validator = run_comprehensive_validation()
        results['validation'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Error in Validation: {e}")
        results['validation'] = f'FAILED: {e}'

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print(" " * 30 + "EXECUTION SUMMARY")
    print("=" * 80)

    for module, status in results.items():
        status_icon = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_icon} {module:30} {status}")

    print("=" * 80)

    # Count outputs
    try:
        output_files = [f for f in os.listdir('outputs') if f.endswith('.png')]
        total_size = sum(os.path.getsize(os.path.join('outputs', f)) for f in output_files)

        print(f"\n📊 Generated Outputs:")
        print(f"   • Total PNG files: {len(output_files)}")
        print(f"   • Total size: {total_size / (1024*1024):.2f} MB")
        print(f"   • Location: ./outputs/")

        print(f"\n📁 Output Files:")
        for i, fname in enumerate(sorted(output_files), 1):
            size_kb = os.path.getsize(os.path.join('outputs', fname)) / 1024
            print(f"   {i:2}. {fname:50} ({size_kb:6.1f} KB)")

    except Exception as e:
        print(f"Error counting outputs: {e}")

    print("\n" + "=" * 80)
    print(" " * 25 + "EXECUTION COMPLETED")
    print("=" * 80)

    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    total_count = len(results)

    if success_count == total_count:
        print(f"\n🎉 All {total_count} modules executed successfully!")
    else:
        print(f"\n⚠ {success_count}/{total_count} modules executed successfully")

    print("\nNext steps:")
    print("  1. Review generated PNG files in './outputs/' directory")
    print("  2. Check validation report for model verification")
    print("  3. Compare results with original research papers")

    return results


if __name__ == "__main__":
    results = main()
