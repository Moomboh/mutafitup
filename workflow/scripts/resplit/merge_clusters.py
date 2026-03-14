"""Merge clustering results from one or more tools via union of similarity edges.

Algorithm:
1. Build a graph where each sequence is a node.
2. For each cluster in each tool, add edges between the representative and
   each member.
3. Find connected components — these are the "union clusters."

Output: ``merged_clusters.tsv`` with columns ``cluster_id``, ``member``.
"""

import statistics
from pathlib import Path

import networkx as nx
import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def _pct(part: int, total: int) -> str:
    """Format a percentage string, safe against zero division."""
    return f"{part / total * 100:.1f}%" if total else "0.0%"


def merge_clusters(tool_paths: dict[str, Path], output_path: Path):
    """Build union graph from one or more clustering tools and output connected-component clusters.

    Parameters
    ----------
    tool_paths:
        Mapping of tool name to cluster TSV path.  Each TSV must have
        columns ``representative`` and ``member``.
    output_path:
        Where to write the merged ``cluster_id, member`` TSV.
    """

    G = nx.Graph()

    for tool_name, tool_path in tool_paths.items():
        df = pd.read_csv(tool_path, sep="\t")
        if "representative" not in df.columns or "member" not in df.columns:
            raise ValueError(
                f"Expected columns 'representative' and 'member' in {tool_path}, "
                f"got: {list(df.columns)}"
            )
        # Convert to strings in case IDs are integers
        df["representative"] = df["representative"].astype(str)
        df["member"] = df["member"].astype(str)

        n_rows = len(df)
        n_reps = df["representative"].nunique()
        n_non_self = int((df["representative"] != df["member"]).sum())

        logger.info(
            "[%s] %d rows, %d unique representatives, %d non-self edges",
            tool_name,
            n_rows,
            n_reps,
            n_non_self,
        )

        # Add all nodes (including singletons where representative == member)
        G.add_nodes_from(df["representative"])
        G.add_nodes_from(df["member"])

        # Add edges between representative and member
        for _, row in df.iterrows():
            if row["representative"] != row["member"]:
                G.add_edge(row["representative"], row["member"])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    logger.info("Union graph: %d nodes, %d edges", n_nodes, n_edges)

    # Find connected components and assign cluster IDs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    component_sizes = []
    for cluster_id, component in enumerate(
        sorted(nx.connected_components(G), key=lambda c: min(c))
    ):
        component_sizes.append(len(component))
        for member in sorted(component):
            rows.append({"cluster_id": cluster_id, "member": member})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, sep="\t", index=False)

    n_clusters = len(component_sizes)
    if n_clusters > 0:
        n_singletons = sum(1 for s in component_sizes if s == 1)
        largest = max(component_sizes)
        median_size = int(statistics.median(component_sizes))
        logger.info(
            "Merged into %d clusters (%d singletons, %s; largest: %d; median size: %d)",
            n_clusters,
            n_singletons,
            _pct(n_singletons, n_clusters),
            largest,
            median_size,
        )
    else:
        logger.info("Merged into 0 clusters")


def main():
    enabled_tools = list(snakemake.params.tools)
    tool_paths = {}
    for tool in enabled_tools:
        tool_paths[tool] = Path(snakemake.input[tool])
    output_path = Path(snakemake.output.merged)

    merge_clusters(tool_paths, output_path)


main()
