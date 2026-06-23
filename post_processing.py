"""
post_processing.py
Domain-knowledge correction layer for materials informatics ML predictions.
"""

from pymatgen.core import Composition


KNOWN_SEMICONDUCTORS = {"Si", "Ge", "GaAs", "SiC", "MoS2", "ZnO", "Cu2O", "CuO"}
KNOWN_INSULATORS = {"NaCl", "MgO", "Al2O3", "SiO2", "BN"}
KNOWN_CORRELATED_OXIDES = {"NiO", "Co3O4", "FeO", "MnO2"}


def correct_formation_energy(formula: str, predicted_energy: float) -> float:
    """
    Corrects formation energy predictions.
    Pure elements should have a formation energy of 0.0 eV/atom.
    """
    try:
        comp = Composition(formula)
        if len(comp.elements) == 1:
            return 0.0
        return predicted_energy
    except Exception:

        return predicted_energy


def correct_electronic_state(formula: str, predicted_state: str, band_gap: float = None) -> tuple:
    """
    Corrects electronic state using hardcoded overrides and chemical heuristics.
    Returns: (corrected_state, rule_applied)
    """

    if band_gap is not None:
        if band_gap == 0:
            return "Metal", "override_bandgap"
        elif band_gap > 3.0:
            return "Insulator", "override_bandgap"
        elif band_gap > 0:
            return "Semiconductor", "override_bandgap"


    if formula in KNOWN_SEMICONDUCTORS:
        return "Semiconductor", "override_known"
    if formula in KNOWN_INSULATORS:
        return "Insulator", "override_known"
    if formula in KNOWN_CORRELATED_OXIDES:
        return "Insulator", "override_known"  


    try:
        comp = Composition(formula)
        elements = comp.elements

        is_all_metals = all(el.is_metal for el in elements)
        has_metal = any(el.is_metal for el in elements)
        has_nonmetal = any(not el.is_metal for el in elements)
        has_tm = any(el.is_transition_metal for el in elements)
        has_oxygen = any(el.symbol == "O" for el in elements)


        if is_all_metals:
            return "Metal", "heuristic_all_metal"


        if has_tm and has_oxygen:
            return "Insulator", "heuristic_tm_oxide"


        if has_metal and has_nonmetal:

            if predicted_state.lower() == "metal":
                return "Insulator", "heuristic_ionic_covalent"
            return predicted_state, "heuristic_confirmed"


        return predicted_state, "ml_baseline"

    except Exception:

        return predicted_state, "ml_baseline"


def estimate_confidence(formula: str, rule_applied: str) -> str:
    """
    Determines prediction confidence based on the source of the prediction
    and the chemical complexity of the material.
    """

    if "override" in rule_applied:
        return "High"
    elif "heuristic" in rule_applied:
        base_confidence = "Medium"
    else:
        base_confidence = "Low"


    try:
        comp = Composition(formula)
        elements = comp.elements

        has_tm = any(el.is_transition_metal for el in elements)
        has_lanthanoid_actinoid = any(el.is_actinoid or el.is_lanthanoid for el in elements)
        is_complex = len(elements) >= 4


        if "override" not in rule_applied:
            if has_lanthanoid_actinoid or is_complex:
                return "Low"
            if has_tm and base_confidence == "Medium":
                return "Low"  

        return base_confidence

    except Exception:
        return "Low"


def apply_correction_pipeline(formula: str, predicted_energy: float, predicted_state: str,
                              band_gap: float = None) -> dict:
    """
    Main wrapper function to process a single material's ML predictions.
    """

    energy_final = correct_formation_energy(formula, predicted_energy)


    state_final, rule_applied = correct_electronic_state(formula, predicted_state, band_gap)


    confidence_final = estimate_confidence(formula, rule_applied)

    return {
        "formula": formula,
        "energy_ev_atom": energy_final,
        "electronic_state": state_final,
        "confidence": confidence_final,
        "_debug_rule_applied": rule_applied  
    }