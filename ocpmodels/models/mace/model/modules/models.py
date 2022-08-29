###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch

try:
    from e3nn import o3

    from experimental.abhshkdz.mace.model.tools.scatter import scatter_sum
except:
    pass

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

from .blocks import (  # Interaction classes
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from .utils import compute_forces


@registry.register_model("mace_v8.15.1")
@registry.register_model("mace_v8.15.2")
@registry.register_model("mace_v8.16.1")
@registry.register_model("mace_v8.16.2")
@registry.register_model("mace_v8.17.1")
@registry.register_model("mace_v8.18.1")
class MACE(BaseModel):
    def __init__(
        self,
        # Unused legacy OCP params.
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        #
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        correlation: int,
        # Defaults from OCP / https://github.com/ACEsuit/mace/blob/main/scripts/run_train.py
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([0.0 for i in range(83)]),
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        max_neighbors: int = 50,
        otf_graph: bool = True,
        use_pbc: bool = True,
        regress_forces: bool = True,
    ):
        super().__init__()
        self.cutoff = self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.max_neighbors = max_neighbors
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces

        # YAML loads them as strings, initialize them as o3.Irreps.
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            # MACE considers a fixed `avg_num_neighbors`. We can either compute
            # this ~fixed statistic for OC20, or compute this value on-the-fly.
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            element_dependent=True,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                element_dependent=True,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    @conditional_grad(torch.enable_grad())
    def forward(self, data):

        # TODO(@abhshkdz): Fit linear references per element from training data.
        #   These are currently initialized to 0.0.

        # OCP prepro boilerplate.
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        num_graphs = data.batch.max() + 1

        # MACE computes forces via gradients.
        pos.requires_grad_(True)

        (
            edge_index,
            D_st,
            distance_vec,
            cell_offsets,
            neighbors,
        ) = self.generate_graph(data)
        idx_s, idx_t = edge_index
        ### OCP prepro ends.

        # Atomic energies
        #
        # Comment(@abhshkdz): `data.node_attrs` is a 1-hot vector for each
        # atomic number. `self.atomic_energies_fn` just matmuls the 1-hot
        # vectors with the list of energies per atomic number, returning the
        # energy per element.
        #
        # For OC20, we initialize these per-element energies to 0.0 since
        # we don't really need linear referencing for adsorption energies?

        atomic_numbers -= 1  # subtract 1 because our embeddings start from 0.
        atomic_numbers_1hot = torch.zeros(
            num_atoms,
            len(self.atomic_energies_fn.atomic_energies),
            device=atomic_numbers.device,
        ).scatter_(1, atomic_numbers.unsqueeze(1), 1.0)

        node_e0 = self.atomic_energies_fn(atomic_numbers_1hot)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(atomic_numbers_1hot)

        # Comment(@abhshkdz): `lengths` here is same as `D_st`, and `vectors` is
        # the same as `distance_vec` but pointing in the opposite direction.
        # vectors, lengths = get_edge_vectors_and_lengths(
        #     positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        # )
        lengths = D_st.view(-1, 1)
        vectors = -distance_vec

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            # print((neighbors / data.natoms).mean(), self.avg_num_neighbors)
            node_feats, sc = interaction(
                node_attrs=atomic_numbers_1hot,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=atomic_numbers_1hot
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data.batch,
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        # Compute forces via autograd.
        forces = compute_forces(
            energy=total_energy, positions=pos, training=self.training
        )

        # return {
        #     "energy": total_energy,
        #     "contributions": contributions,
        #     "forces": forces,
        # }
        return total_energy, forces
