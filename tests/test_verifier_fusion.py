"""Tests du contrôle de fusion, sur de minuscules faux modèles.

Un contrôle qui ne sait pas échouer ne prouve rien : ces tests fabriquent une fusion correcte, puis la
cassent de chaque façon que le contrôle doit détecter.

Lancer depuis la racine du repo :
    uv run python -m pytest
"""

import json

import pytest

# torch et MLX sont dans le groupe `ml`, que la CI n'installe pas (plusieurs Go) : on ignore le fichier
# au lieu de faire échouer toute la collecte.
torch = pytest.importorskip("torch")
pytest.importorskip("mlx.core")

from safetensors.torch import save_file

from src.finetuning.verifier_fusion import calculer_fusion, controler_fusion

SCALE = 20.0
CIBLE = "model.layers.1.mlp.up_proj.weight"
HORS_CIBLE = "model.layers.0.mlp.up_proj.weight"


def fusion_correcte() -> tuple[dict, dict, dict]:
    """Renvoie (base, fusionné, adaptateur) cohérents entre eux."""
    torch.manual_seed(0)
    base = {
        CIBLE: torch.randn(6, 4).to(torch.bfloat16),
        HORS_CIBLE: torch.randn(6, 4).to(torch.bfloat16),
    }
    # Convention de mlx-lm : lora_a est (entrée, rang) et lora_b (rang, sortie).
    lora_a = torch.randn(4, 2)
    lora_b = torch.randn(2, 6)
    fusionne = {CIBLE: calculer_fusion(base[CIBLE], lora_a, lora_b, SCALE), HORS_CIBLE: base[HORS_CIBLE].clone()}
    adaptateur = {"model.layers.1.mlp.up_proj.lora_a": lora_a, "model.layers.1.mlp.up_proj.lora_b": lora_b}
    return base, fusionne, adaptateur


def lancer_controle(tmp_path, base, fusionne, adaptateur):
    for nom in ("base", "fusionne", "adaptateur"):
        (tmp_path / nom).mkdir()
    save_file(base, tmp_path / "base" / "model.safetensors")
    save_file(fusionne, tmp_path / "fusionne" / "model.safetensors")
    save_file(adaptateur, tmp_path / "adaptateur" / "adapters.safetensors")
    config = {"lora_parameters": {"rank": 2, "scale": SCALE}}
    (tmp_path / "adaptateur" / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return controler_fusion(tmp_path / "base", tmp_path / "fusionne", tmp_path / "adaptateur")


def test_fusion_correcte_acceptee(tmp_path):
    bilan = lancer_controle(tmp_path, *fusion_correcte())
    assert bilan.erreurs == []
    assert bilan.n_cibles == 1
    assert bilan.n_inchanges == 1


def _modifier_hors_cible(fusionne):
    fusionne[HORS_CIBLE] = fusionne[HORS_CIBLE] + 1


def _ignorer_adaptateur(fusionne):
    fusionne[CIBLE] = fusionne[HORS_CIBLE].clone()


def _copier_base_sur_cible(base, fusionne):
    fusionne[CIBLE] = base[CIBLE].clone()


def _supprimer_tenseur(fusionne):
    del fusionne[HORS_CIBLE]


def _ajouter_tenseur(fusionne):
    fusionne["model.layers.9.en_trop.weight"] = torch.zeros(2, 2).to(torch.bfloat16)


def _changer_precision(fusionne):
    fusionne[HORS_CIBLE] = fusionne[HORS_CIBLE].to(torch.float16)


@pytest.mark.parametrize(
    "casser",
    [_modifier_hors_cible, _ignorer_adaptateur, _supprimer_tenseur, _ajouter_tenseur, _changer_precision],
)
def test_fusion_cassee_detectee(tmp_path, casser):
    base, fusionne, adaptateur = fusion_correcte()
    casser(fusionne)
    assert lancer_controle(tmp_path, base, fusionne, adaptateur).erreurs


def test_adaptateur_non_applique_detecte(tmp_path):
    base, fusionne, adaptateur = fusion_correcte()
    _copier_base_sur_cible(base, fusionne)
    erreurs = lancer_controle(tmp_path, base, fusionne, adaptateur).erreurs
    assert any("identique à la base" in erreur for erreur in erreurs)


def test_mauvais_scale_detecte(tmp_path):
    base, fusionne, adaptateur = fusion_correcte()
    # Mauvais scale : les poids changent, donc seule la comparaison au calcul attendu peut le voir.
    fusionne[CIBLE] = calculer_fusion(
        base[CIBLE],
        adaptateur["model.layers.1.mlp.up_proj.lora_a"],
        adaptateur["model.layers.1.mlp.up_proj.lora_b"],
        SCALE / 2,
    )
    erreurs = lancer_controle(tmp_path, base, fusionne, adaptateur).erreurs
    assert any("calcul attendu" in erreur for erreur in erreurs)
