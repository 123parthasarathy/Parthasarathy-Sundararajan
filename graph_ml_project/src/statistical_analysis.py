"""
Advanced Statistical Analysis for Graphs
Combining classical graph theory with modern statistical methods
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class GraphStatistics:
    """Comprehensive statistical analysis of graphs"""

    def __init__(self, G):
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()

    def compute_centrality_measures(self):
        """Compute various centrality measures"""
        centralities = {}

        # Degree centrality
        centralities['degree'] = nx.degree_centrality(self.G)

        # Betweenness centrality
        centralities['betweenness'] = nx.betweenness_centrality(self.G)

        # Closeness centrality
        if nx.is_connected(self.G):
            centralities['closeness'] = nx.closeness_centrality(self.G)
        else:
            # For disconnected graphs
            centralities['closeness'] = {n: 0 for n in self.G.nodes()}

        # Eigenvector centrality
        try:
            centralities['eigenvector'] = nx.eigenvector_centrality(self.G, max_iter=1000)
        except:
            centralities['eigenvector'] = {n: 0 for n in self.G.nodes()}

        # PageRank
        centralities['pagerank'] = nx.pagerank(self.G)

        return centralities

    def compute_structural_properties(self):
        """Compute structural properties"""
        properties = {}

        # Basic properties
        properties['num_nodes'] = self.n_nodes
        properties['num_edges'] = self.n_edges
        properties['density'] = nx.density(self.G)
        properties['is_connected'] = nx.is_connected(self.G)

        # Clustering
        properties['avg_clustering'] = nx.average_clustering(self.G)
        properties['transitivity'] = nx.transitivity(self.G)

        # Degree statistics
        degrees = [d for n, d in self.G.degree()]
        properties['avg_degree'] = np.mean(degrees)
        properties['std_degree'] = np.std(degrees)
        properties['max_degree'] = np.max(degrees)
        properties['min_degree'] = np.min(degrees)

        # Assortativity
        try:
            properties['assortativity'] = nx.degree_assortativity_coefficient(self.G)
        except:
            properties['assortativity'] = 0

        # Components
        if nx.is_connected(self.G):
            properties['num_components'] = 1
            properties['diameter'] = nx.diameter(self.G)
            properties['avg_shortest_path'] = nx.average_shortest_path_length(self.G)
        else:
            components = list(nx.connected_components(self.G))
            properties['num_components'] = len(components)
            properties['diameter'] = max([nx.diameter(self.G.subgraph(c)) for c in components if len(c) > 1] or [0])
            properties['avg_shortest_path'] = 0

        return properties

    def compute_degree_distribution(self):
        """Compute and analyze degree distribution"""
        degrees = [d for n, d in self.G.degree()]
        degree_counts = pd.Series(degrees).value_counts().sort_index()

        # Fit power law (scale-free test)
        unique_degrees = degree_counts.index.values
        counts = degree_counts.values

        if len(unique_degrees) > 2:
            # Log-log regression for power law
            log_degrees = np.log(unique_degrees[unique_degrees > 0])
            log_counts = np.log(counts[unique_degrees > 0])

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_counts)

            powerlaw_params = {
                'exponent': -slope,
                'r_squared': r_value**2,
                'p_value': p_value
            }
        else:
            powerlaw_params = None

        return {
            'degrees': degrees,
            'degree_counts': degree_counts,
            'powerlaw_params': powerlaw_params
        }

    def compute_community_structure(self):
        """Detect and analyze community structure"""
        # Greedy modularity communities
        communities = list(nx.community.greedy_modularity_communities(self.G))
        modularity = nx.community.modularity(self.G, communities)

        # Community sizes
        community_sizes = [len(c) for c in communities]

        return {
            'num_communities': len(communities),
            'modularity': modularity,
            'community_sizes': community_sizes,
            'communities': communities
        }

    def compute_motif_counts(self):
        """Count graph motifs (small subgraph patterns)"""
        motifs = {}

        # Triangles
        motifs['triangles'] = sum(nx.triangles(self.G).values()) // 3

        # 4-cliques
        cliques = list(nx.find_cliques(self.G))
        motifs['4_cliques'] = len([c for c in cliques if len(c) == 4])
        motifs['max_clique_size'] = max([len(c) for c in cliques]) if cliques else 0

        return motifs


def visualize_degree_distribution(degree_data, save_path='outputs/degree_distribution.png', dpi=100):
    """Visualize degree distribution with power-law fit"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    degrees = degree_data['degrees']
    degree_counts = degree_data['degree_counts']

    # Linear scale
    axes[0].bar(degree_counts.index, degree_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Log-log scale (power-law test)
    unique_degrees = degree_counts.index.values
    counts = degree_counts.values

    axes[1].scatter(unique_degrees, counts, s=100, alpha=0.6, color='steelblue', edgecolors='black')

    # Add power-law fit if available
    if degree_data['powerlaw_params'] is not None:
        params = degree_data['powerlaw_params']
        x_fit = np.linspace(unique_degrees.min(), unique_degrees.max(), 100)
        y_fit = np.exp(np.log(x_fit) * (-params['exponent']))
        y_fit = y_fit * counts.max() / y_fit.max()  # Normalize

        axes[1].plot(x_fit, y_fit, 'r--', linewidth=2, label=f"Power-law fit\nγ = {params['exponent']:.2f}\nR² = {params['r_squared']:.3f}")
        axes[1].legend(fontsize=10)

    axes[1].set_xlabel('Degree (log scale)', fontsize=12)
    axes[1].set_ylabel('Count (log scale)', fontsize=12)
    axes[1].set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_centrality_distribution(centralities, save_path='outputs/centrality_distribution.png', dpi=100):
    """Visualize distribution of centrality measures"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    centrality_types = ['degree', 'betweenness', 'closeness', 'pagerank']
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

    for ax, cent_type, color in zip(axes, centrality_types, colors):
        if cent_type in centralities:
            values = list(centralities[cent_type].values())

            # Histogram
            ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor='black')

            # Add statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

            ax.set_xlabel(f'{cent_type.capitalize()} Centrality', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{cent_type.capitalize()} Centrality Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_centrality_correlation(centralities, save_path='outputs/centrality_correlation.png', dpi=100):
    """Visualize correlation between centrality measures"""
    # Create dataframe
    df = pd.DataFrame(centralities)

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.3f', ax=ax)

    ax.set_title('Centrality Measures Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_properties_summary(properties, save_path='outputs/graph_properties.png', dpi=100):
    """Visualize summary of graph properties"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Basic statistics
    basic_props = {
        'Nodes': properties['num_nodes'],
        'Edges': properties['num_edges'],
        'Density': properties['density'],
        'Avg Degree': properties['avg_degree'],
        'Avg Clustering': properties['avg_clustering']
    }

    axes[0, 0].axis('off')
    table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in basic_props.items()]
    table = axes[0, 0].table(cellText=table_data, colLabels=['Property', 'Value'],
                            cellLoc='left', loc='center',
                            colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for i in range(len(table_data) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')

    axes[0, 0].set_title('Basic Graph Properties', fontsize=12, fontweight='bold', pad=10)

    # 2. Degree statistics
    degree_stats = {
        'Mean': properties['avg_degree'],
        'Std Dev': properties['std_degree'],
        'Min': properties['min_degree'],
        'Max': properties['max_degree']
    }
    axes[0, 1].bar(degree_stats.keys(), degree_stats.values(), color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Value', fontsize=11)
    axes[0, 1].set_title('Degree Statistics', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Network metrics
    network_metrics = {
        'Clustering': properties['avg_clustering'],
        'Transitivity': properties['transitivity'],
        'Assortativity': properties['assortativity'],
        'Density': properties['density']
    }

    axes[1, 0].barh(list(network_metrics.keys()), list(network_metrics.values()),
                   color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Value', fontsize=11)
    axes[1, 0].set_title('Network Metrics', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Components info
    if properties['is_connected']:
        comp_text = f"Graph is CONNECTED\n\nDiameter: {properties['diameter']}\nAvg Path Length: {properties['avg_shortest_path']:.3f}"
        color = 'lightgreen'
    else:
        comp_text = f"Graph is DISCONNECTED\n\nComponents: {properties['num_components']}\nDiameter: {properties['diameter']}"
        color = 'lightyellow'

    axes[1, 1].text(0.5, 0.5, comp_text, ha='center', va='center',
                   fontsize=14, bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Connectivity', fontsize=12, fontweight='bold')

    plt.suptitle('Graph Statistical Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def demonstrate_statistical_analysis():
    """Demonstrate comprehensive statistical analysis"""
    print("\n" + "=" * 70)
    print("ADVANCED STATISTICAL ANALYSIS FOR GRAPHS")
    print("=" * 70)

    # Create sample graphs
    graphs = [
        ("Karate Club", nx.karate_club_graph()),
        ("Barabasi-Albert", nx.barabasi_albert_graph(100, 3, seed=42)),
        ("Watts-Strogatz", nx.watts_strogatz_graph(100, 6, 0.3, seed=42))
    ]

    for name, G in graphs:
        print(f"\n{'─' * 70}")
        print(f"Analyzing: {name}")
        print(f"{'─' * 70}")

        stats = GraphStatistics(G)

        # Compute all statistics
        properties = stats.compute_structural_properties()
        centralities = stats.compute_centrality_measures()
        degree_data = stats.compute_degree_distribution()
        communities = stats.compute_community_structure()
        motifs = stats.compute_motif_counts()

        print(f"\n📊 Structural Properties:")
        print(f"   Nodes: {properties['num_nodes']}, Edges: {properties['num_edges']}")
        print(f"   Density: {properties['density']:.4f}")
        print(f"   Avg Clustering: {properties['avg_clustering']:.4f}")
        print(f"   Assortativity: {properties['assortativity']:.4f}")

        print(f"\n🔍 Community Structure:")
        print(f"   Communities: {communities['num_communities']}")
        print(f"   Modularity: {communities['modularity']:.4f}")

        print(f"\n🔺 Motif Counts:")
        print(f"   Triangles: {motifs['triangles']}")
        print(f"   Max Clique Size: {motifs['max_clique_size']}")

        # Visualizations
        name_clean = name.replace(" ", "_")

        save_path = visualize_degree_distribution(degree_data,
                                                  f'outputs/degree_dist_{name_clean}.png')
        print(f"\n✓ Saved degree distribution: {save_path}")

        save_path = visualize_centrality_distribution(centralities,
                                                      f'outputs/centrality_dist_{name_clean}.png')
        print(f"✓ Saved centrality distribution: {save_path}")

        save_path = visualize_centrality_correlation(centralities,
                                                     f'outputs/centrality_corr_{name_clean}.png')
        print(f"✓ Saved centrality correlation: {save_path}")

        save_path = visualize_properties_summary(properties,
                                                f'outputs/properties_{name_clean}.png')
        print(f"✓ Saved properties summary: {save_path}")

    print("\n" + "=" * 70)
    print("Statistical Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_statistical_analysis()
