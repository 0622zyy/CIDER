from types import SimpleNamespace

import torch
from torch import nn

from src.methods import CEDR, LCI
from src.methods.cedr import EvidenceDecoupling, MemoryReliabilityRefinement
from src.methods._internal.tmr import build_tail_memories


class FakeModel(nn.Module):
    def __init__(self, classes: int = 5, channels: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, channels, 1)
        self.classifier = nn.Linear(channels, classes)
        layer = SimpleNamespace(last_attn_weights=None)
        self.ml_decoder = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))

    def forward_features(self, images):
        return self.projection(images)

    def forward_head(self, feature_map):
        spatial = feature_map.square().mean(dim=1).flatten(1)
        attention = torch.softmax(spatial, dim=1)
        self.ml_decoder.decoder.layers[-1].last_attn_weights = attention[:, None].repeat(
            1, self.classifier.out_features, 1
        )
        return self.classifier(feature_map.mean(dim=(2, 3)))


def build_modules():
    tail_indices = torch.tensor([3, 4])
    decoupling = EvidenceDecoupling(
        num_classes=5,
        tail_indices=tail_indices,
        confounder_indices=torch.tensor([[0, 1], [1, 0]]),
        confounder_weights=torch.ones(2, 2),
        tail_positive_weights=torch.ones(2),
        feature_dim=8,
        hidden_dim=4,
        dropout=0.0,
        confounder_strength=1.0,
        private_residual_floor=0.2,
        maximum_logit_correction=4.0,
        initial_gate_logit=-2.0,
        negative_loss_weight=0.2,
        ranking_margin=1.0,
        evidence_margin=0.5,
        hard_negative_count=2,
        tail_bce_weight=1.0,
        ranking_weight=1.0,
        evidence_margin_weight=0.5,
        context_suppression_weight=0.2,
        attention_overlap_weight=0.2,
        feature_orthogonality_weight=0.1,
        correction_regularization_weight=0.01,
    )
    features = torch.randn(12, 2, 8)
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]] + [[0, 0]] * 8).float()
    memory = build_tail_memories(
        features,
        targets,
        torch.randn(12, 2),
        positive_prototypes=2,
        negative_prototypes=2,
        hard_negative_pool=5,
    )
    refinement = MemoryReliabilityRefinement(
        num_classes=5,
        tail_indices=tail_indices,
        positive_memory=memory["positive_memory"],
        negative_memory=memory["negative_memory"],
        positive_valid=memory["positive_valid"],
        negative_valid=memory["negative_valid"],
        confounder_indices=torch.tensor([[0, 1], [1, 0]]),
    )
    return decoupling, refinement


def test_public_api_and_lci_shape() -> None:
    import src.methods as methods

    assert methods.__all__ == ["LCI", "CEDR"]
    model = FakeModel()
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor(
        [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 1, 0, 0]]
    ).float()
    lci = LCI(
        num_classes=5,
        class_split={"head": [0, 1], "middle": [2], "tail": [3, 4]},
        class_frequency=torch.tensor([4, 4, 3, 2, 2]),
        core_topk_ratio=0.25,
        necessity_margin=1.0,
        swap_positive_weight=0.5,
        swap_invariance_weight=0.2,
        core_necessity_weight=0.2,
        auxiliary_to_classification_cap=1.0,
        head_sample_weight=0.1,
        middle_sample_weight=0.5,
        tail_sample_weight=1.0,
    )
    result = lci(model, images, targets, torch.tensor(1.0), 1.0)
    assert result["logits"].shape == (4, 5)
    assert result["feature_map"].shape == (4, 8, 4, 4)


def test_cedr_connection_mapping_and_roundtrip() -> None:
    torch.manual_seed(7)
    model = FakeModel()
    images = torch.randn(4, 3, 4, 4)
    feature = model.forward_features(images)
    baseline = model.forward_head(feature)
    attention = model.ml_decoder.decoder.layers[-1].last_attn_weights
    decoupling, refinement = build_modules()
    with torch.no_grad():
        refinement.class_bias.copy_(torch.tensor([0.3, -0.4]))
    cedr = CEDR(decoupling, refinement, residual_alpha=0.8).eval()
    result = cedr(feature, attention, baseline)
    assert result["logits"].shape == baseline.shape
    assert torch.equal(cedr.tail_indices, torch.tensor([3, 4]))
    assert torch.equal(result["logits"][:, :3], baseline[:, :3])
    assert result["private_feature"].shape == (4, 2, 8)
    assert result["shared_feature"].shape == (4, 2, 8)
    state = cedr.state_dict()
    decoupling2, refinement2 = build_modules()
    cedr2 = CEDR(decoupling2, refinement2, residual_alpha=0.1).eval()
    cedr2.load_state_dict(state, strict=True)
    result2 = cedr2(feature, attention, baseline)
    assert torch.equal(result["logits"], result2["logits"])
    assert torch.equal(result["memory_residual"], result2["memory_residual"])
    assert float(cedr2.residual_alpha) == float(cedr.residual_alpha)


if __name__ == "__main__":
    test_public_api_and_lci_shape()
    test_cedr_connection_mapping_and_roundtrip()
    print("two_module_cedr_tests_ok")
