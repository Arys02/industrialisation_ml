from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterParameters:
    u_clusters: int
    p_clusters: int
    seed: int
