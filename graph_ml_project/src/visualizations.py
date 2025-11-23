"""
Comprehensive Graph Visualizations and Architecture Diagrams
Advanced network visualization techniques for ML and statistical analysis
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


def visualize_network_layout(G, title='Network Graph', save_path='outputs/network.png',
                             node_color_attr=None, dpi=100):
    """Visualize network with multiple layout algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    layouts = [
        ('Spring Layout', nx.spring_layout),
        ('Kamada-Kawai Layout', nx.kamada_kawai_layout),
        ('Circular Layout', nx.circular_layout),
        ('Spectral Layout', nx.spectral_layout)
    ]

    # Node colors
    if node_color_attr and node_color_attr in nx.get_node_attributes(G, node_color_attr):
        node_colors = [nx.get_node_attributes(G, node_color_attr)[node] for node in G.nodes()]
    else:
        node_colors = [G.degree(node) for node in G.nodes()]

    for ax, (layout_name, layout_func) in zip(axes, layouts):
        try:
            pos = layout_func(G)

            # Draw network
            nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                          node_size=300, cmap='viridis',
                                          alpha=0.8, ax=ax, edgecolors='black', linewidths=1.5)

            nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, edge_color='gray', ax=ax)

            # Add colorbar
            plt.colorbar(nodes, ax=ax, label='Degree' if not node_color_attr else node_color_attr)

            ax.set_title(layout_name, fontsize=14, fontweight='bold')
            ax.axis('off')
        except:
            ax.text(0.5, 0.5, f'{layout_name}\nNot applicable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_community_structure(G, communities, save_path='outputs/communities.png', dpi=100):
    """Visualize community structure with different colors"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Assign colors to communities
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    node_colors = [node_to_community.get(node, 0) for node in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500,
                          cmap='tab20', alpha=0.9, ax=ax, edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, edge_color='gray', ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(f'Community Structure ({len(communities)} communities)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_centrality_map(G, centrality, centrality_name='Centrality',
                             save_path='outputs/centrality_map.png', dpi=100):
    """Visualize node centrality as network map"""
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Node sizes based on centrality
    node_sizes = [centrality.get(node, 0) * 3000 for node in G.nodes()]
    node_colors = [centrality.get(node, 0) for node in G.nodes()]

    # Draw
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                   cmap='YlOrRd', alpha=0.9, ax=ax,
                                   edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax)

    # Top 5 nodes labels
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_labels = {node: f"{node}\n({score:.3f})" for node, score in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=9,
                           font_weight='bold', ax=ax)

    plt.colorbar(nodes, ax=ax, label=f'{centrality_name} Score')

    ax.set_title(f'{centrality_name} Visualization', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def create_ml_architecture_diagram(save_path='outputs/architecture_diagram.png', dpi=100):
    """Create ML/GNN architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    input_color = '#E3F2FD'
    conv_color = '#BBDEFB'
    pool_color = '#90CAF9'
    fc_color = '#64B5F6'
    output_color = '#42A5F5'

    # Title
    ax.text(5, 9.5, 'Graph Neural Network Architecture', ha='center', va='center',
           fontsize=20, fontweight='bold')

    # Layer positions
    layers = [
        (0.5, 5, 'Input\nGraph', input_color, 1.2, 2),
        (2.2, 5, 'GCN\nLayer 1', conv_color, 1.2, 2),
        (3.9, 5, 'GCN\nLayer 2', conv_color, 1.2, 2),
        (5.6, 5, 'GCN\nLayer 3', conv_color, 1.2, 2),
        (7.3, 5, 'Global\nPooling', pool_color, 1.2, 2),
        (9.0, 5, 'Output\nLayer', output_color, 1.2, 2),
    ]

    # Draw layers
    for x, y, label, color, width, height in layers:
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.1", edgecolor='black',
                            facecolor=color, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw arrows
    for i in range(len(layers) - 1):
        x1, y1 = layers[i][0] + layers[i][4]/2, layers[i][1]
        x2, y2 = layers[i+1][0] - layers[i+1][4]/2, layers[i+1][1]

        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=30,
                              linewidth=3, color='black', alpha=0.7)
        ax.add_patch(arrow)

    # Add annotations
    annotations = [
        (2.2, 3.5, 'h¹ = σ(ÂXW¹)', 10),
        (3.9, 3.5, 'h² = σ(Âh¹W²)', 10),
        (5.6, 3.5, 'h³ = σ(Âh²W³)', 10),
        (7.3, 3.5, 'h_graph = AGG(h³)', 9),
    ]

    for x, y, text, fontsize in annotations:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend
    legend_y = 1.5
    ax.text(1, legend_y, 'Legend:', fontsize=12, fontweight='bold')
    ax.text(1, legend_y - 0.4, '• Â: Normalized adjacency matrix', fontsize=9)
    ax.text(1, legend_y - 0.7, '• X: Input node features', fontsize=9)
    ax.text(1, legend_y - 1.0, '• W: Learnable weight matrices', fontsize=9)
    ax.text(1, legend_y - 1.3, '• σ: Activation function (ReLU)', fontsize=9)
    ax.text(1, legend_y - 1.6, '• AGG: Aggregation (mean/sum pooling)', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def create_topological_architecture(save_path='outputs/topological_architecture.png', dpi=100):
    """Create architecture diagram for topological GNN"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.3, 'Topological Graph Neural Network Architecture', ha='center', va='center',
           fontsize=18, fontweight='bold')
    ax.text(6, 8.8, 'Based on: Line Graph Vietoris-Rips Persistence Diagram (JMLR 2024)',
           ha='center', va='center', fontsize=11, style='italic')

    # Two parallel branches
    # Branch 1: GNN Branch (top)
    gnn_y = 6.5
    gnn_boxes = [
        (1, gnn_y, 'Input\nGraph', '#E8F5E9', 1.3),
        (3, gnn_y, 'GCN\nLayers', '#C8E6C9', 1.3),
        (5, gnn_y, 'Graph\nEmbedding', '#A5D6A7', 1.3),
    ]

    for x, y, label, color, width in gnn_boxes:
        box = FancyBboxPatch((x - width/2, y - 0.6), width, 1.2,
                            boxstyle="round,pad=0.08", edgecolor='black',
                            facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Branch 2: Topological Branch (bottom)
    topo_y = 4
    topo_boxes = [
        (1, topo_y, 'Input\nGraph', '#FFF3E0', 1.3),
        (3, topo_y, 'Persistence\nDiagram', '#FFE0B2', 1.3),
        (5, topo_y, 'Topological\nFeatures', '#FFCC80', 1.3),
    ]

    for x, y, label, color, width in topo_boxes:
        box = FancyBboxPatch((x - width/2, y - 0.6), width, 1.2,
                            boxstyle="round,pad=0.08", edgecolor='black',
                            facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Fusion layer
    fusion_x, fusion_y = 7.5, 5.25
    fusion_box = FancyBboxPatch((fusion_x - 1, fusion_y - 0.8), 2, 1.6,
                               boxstyle="round,pad=0.1", edgecolor='black',
                               facecolor='#E1BEE7', linewidth=2.5)
    ax.add_patch(fusion_box)
    ax.text(fusion_x, fusion_y, 'Fusion\nLayer', ha='center', va='center',
           fontsize=11, fontweight='bold')

    # Output
    output_x, output_y = 10.5, 5.25
    output_box = FancyBboxPatch((output_x - 0.8, output_y - 0.6), 1.6, 1.2,
                               boxstyle="round,pad=0.08", edgecolor='black',
                               facecolor='#CE93D8', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(output_x, output_y, 'Prediction', ha='center', va='center',
           fontsize=11, fontweight='bold')

    # Arrows - GNN branch
    for i in range(len(gnn_boxes) - 1):
        x1, x2 = gnn_boxes[i][0] + gnn_boxes[i][4]/2, gnn_boxes[i+1][0] - gnn_boxes[i+1][4]/2
        arrow = FancyArrowPatch((x1, gnn_y), (x2, gnn_y), arrowstyle='->', mutation_scale=25,
                              linewidth=2.5, color='darkgreen')
        ax.add_patch(arrow)

    # Arrows - Topo branch
    for i in range(len(topo_boxes) - 1):
        x1, x2 = topo_boxes[i][0] + topo_boxes[i][4]/2, topo_boxes[i+1][0] - topo_boxes[i+1][4]/2
        arrow = FancyArrowPatch((x1, topo_y), (x2, topo_y), arrowstyle='->', mutation_scale=25,
                              linewidth=2.5, color='darkorange')
        ax.add_patch(arrow)

    # Arrows to fusion
    arrow_gnn = FancyArrowPatch((5 + 0.65, gnn_y), (fusion_x - 1, fusion_y + 0.4),
                                arrowstyle='->', mutation_scale=25, linewidth=2.5, color='darkgreen')
    ax.add_patch(arrow_gnn)

    arrow_topo = FancyArrowPatch((5 + 0.65, topo_y), (fusion_x - 1, fusion_y - 0.4),
                                arrowstyle='->', mutation_scale=25, linewidth=2.5, color='darkorange')
    ax.add_patch(arrow_topo)

    # Arrow to output
    arrow_out = FancyArrowPatch((fusion_x + 1, fusion_y), (output_x - 0.8, output_y),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5, color='purple')
    ax.add_patch(arrow_out)

    # Add labels
    ax.text(3, gnn_y + 1.2, 'Graph Structure Branch', ha='center', fontsize=11,
           fontweight='bold', color='darkgreen')
    ax.text(3, topo_y - 1.2, 'Topological Feature Branch', ha='center', fontsize=11,
           fontweight='bold', color='darkorange')

    # Add formulas
    ax.text(3, topo_y - 1.8, 'PD = Ripser(L(G))', ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(5, topo_y - 1.8, 'f_topo = Stats(PD)', ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def create_equivariant_architecture(save_path='outputs/equivariant_architecture.png', dpi=100):
    """Create architecture diagram for equivariant GNN"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.3, 'E(n) Equivariant Graph Neural Network', ha='center', va='center',
           fontsize=18, fontweight='bold')
    ax.text(5, 8.8, 'Preserves Geometric Symmetries: Rotations, Translations, Reflections',
           ha='center', va='center', fontsize=11, style='italic')

    # Input
    input_box = FancyBboxPatch((0.3, 4.5), 1.4, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(1, 5.5, 'Input', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, 5.2, 'h: Features', ha='center', fontsize=9)
    ax.text(1, 4.9, 'x: Positions', ha='center', fontsize=9)

    # EGNN Layers
    layer_positions = [(2.8, 5.25), (4.8, 5.25), (6.8, 5.25)]
    for i, (x, y) in enumerate(layer_positions):
        layer_box = FancyBboxPatch((x - 0.8, y - 0.8), 1.6, 1.6,
                                  boxstyle="round,pad=0.1", edgecolor='black',
                                  facecolor='#E1F5FE', linewidth=2.5)
        ax.add_patch(layer_box)
        ax.text(x, y + 0.3, f'EGNN Layer {i+1}', ha='center', fontsize=10, fontweight='bold')

        # Show equivariant operations
        ax.text(x, y - 0.05, 'h\' = φ_h(h, m)', ha='center', fontsize=8, style='italic')
        ax.text(x, y - 0.35, 'x\' = x + φ_x(m)⊙Δx', ha='center', fontsize=8, style='italic')

    # Output
    output_box = FancyBboxPatch((8.3, 4.5), 1.4, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E8F5E9', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(9, 5.5, 'Output', ha='center', fontsize=11, fontweight='bold')
    ax.text(9, 5.2, 'Invariant', ha='center', fontsize=9)
    ax.text(9, 4.9, 'Prediction', ha='center', fontsize=9)

    # Arrows
    arrow1 = FancyArrowPatch((1.7, 5.25), (2.0, 5.25), arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='black')
    ax.add_patch(arrow1)

    for i in range(len(layer_positions) - 1):
        x1 = layer_positions[i][0] + 0.8
        x2 = layer_positions[i+1][0] - 0.8
        arrow = FancyArrowPatch((x1, 5.25), (x2, 5.25), arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color='black')
        ax.add_patch(arrow)

    arrow_final = FancyArrowPatch((7.6, 5.25), (8.3, 5.25), arrowstyle='->', mutation_scale=25,
                                 linewidth=2.5, color='black')
    ax.add_patch(arrow_final)

    # Symmetry illustration
    sym_y = 2.5
    ax.text(5, 3.5, 'Equivariance Property:', ha='center', fontsize=12, fontweight='bold')

    # Rotation example
    ax.text(2.5, sym_y, 'Input: x', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    ax.text(4.5, sym_y, 'Rotate: R·x', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    ax.text(7, sym_y, 'f(R·x) = R·f(x)', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#C8E6C9'))

    # Translation example
    ax.text(2.5, sym_y - 0.7, 'Input: x', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    ax.text(4.5, sym_y - 0.7, 'Translate: x+t', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    ax.text(7, sym_y - 0.7, 'f(x+t) = f(x)+t', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#C8E6C9'))

    # Legend
    legend_box = FancyBboxPatch((0.2, 0.2), 3.5, 1.2, boxstyle="round,pad=0.1",
                               edgecolor='gray', facecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(legend_box)
    ax.text(0.4, 1.15, 'Legend:', fontsize=10, fontweight='bold')
    ax.text(0.4, 0.85, '• h: Node scalar features (invariant)', fontsize=8)
    ax.text(0.4, 0.6, '• x: Node positions (equivariant)', fontsize=8)
    ax.text(0.4, 0.35, '• φ: Learnable equivariant functions', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def demonstrate_visualizations():
    """Demonstrate all visualization capabilities"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE GRAPH VISUALIZATIONS")
    print("=" * 70)

    # Create sample graphs
    print("\nGenerating sample networks...")
    G_karate = nx.karate_club_graph()
    G_ba = nx.barabasi_albert_graph(80, 3, seed=42)

    # 1. Network layouts
    print("\n📊 Creating network layout visualizations...")
    save_path = visualize_network_layout(G_karate, 'Karate Club Network - Multiple Layouts',
                                        'outputs/network_layouts.png')
    print(f"✓ Saved: {save_path}")

    # 2. Community structure
    print("\n🔍 Detecting and visualizing communities...")
    communities = list(nx.community.greedy_modularity_communities(G_karate))
    save_path = visualize_community_structure(G_karate, communities,
                                              'outputs/community_structure.png')
    print(f"✓ Saved: {save_path}")

    # 3. Centrality visualization
    print("\n⭐ Visualizing node centrality...")
    pagerank = nx.pagerank(G_ba)
    save_path = visualize_centrality_map(G_ba, pagerank, 'PageRank',
                                        'outputs/pagerank_map.png')
    print(f"✓ Saved: {save_path}")

    # 4. Architecture diagrams
    print("\n🏗️  Creating ML architecture diagrams...")
    save_path = create_ml_architecture_diagram()
    print(f"✓ Saved: {save_path}")

    save_path = create_topological_architecture()
    print(f"✓ Saved: {save_path}")

    save_path = create_equivariant_architecture()
    print(f"✓ Saved: {save_path}")

    print("\n" + "=" * 70)
    print("All visualizations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_visualizations()
