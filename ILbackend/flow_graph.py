import matplotlib.pyplot as plt
import networkx as nx

edges = [
    ("Prompt", "Freeform"),
    ("Prompt", "Code Detected"),
    ("Freeform", "research_agent"),
    ("Code Detected", "Single Code Flow"),
    ("Code Detected", "Compatibility Flow"),
    ("research_agent", "LLM Reply"),
    ("Single Code Flow", "process_question"),
    ("Compatibility Flow", "process_question"),
    ("process_question", "LLM Reply")
]

# Color mapping for node paths
node_colors = {
    "Prompt": "lightgray",
    "Freeform": "mediumorchid",
    "research_agent": "mediumorchid",
    "Code Detected": "lightgray",
    "Single Code Flow": "mediumseagreen",
    "Compatibility Flow": "orange",
    "process_question": "gold",
    "LLM Reply": "deepskyblue"
}

# Function to compute size based on text length
def calc_size(label):
    return max(1800, len(label) * 450)

# Node sizes auto-scaled for label visibility
sizes = {node: calc_size(node) for node in node_colors.keys()}

# Build graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Position setup (flip Single Code and Freeform sides)
pos = {
    "Prompt": (0, 0),
    "Freeform": (-2, -1.5),
    "Code Detected": (2, -1.5),
    "research_agent": (-2, -3),
    "Single Code Flow": (2.5, -3),
    "Compatibility Flow": (1, -3),
    "process_question": (1.5, -4.5),
    "LLM Reply": (0, -6)
}

plt.figure(figsize=(12, 8))
# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=25, width=2.5, edge_color='black')

# Draw nodes
for node in G.nodes():
    nx.draw_networkx_nodes(
        G, pos, nodelist=[node], node_size=sizes[node],
        node_color=node_colors[node], edgecolors='black', alpha=0.9
    )

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

plt.title("Clean Decision Flow with Color-coded Paths", fontsize=16, pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()



