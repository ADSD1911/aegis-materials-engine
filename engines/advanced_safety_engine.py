import re
from pymatgen.core import Composition


class AdvancedSafetyEngine:
    """
    A context-aware, chemical reasoning engine for evaluating
    environmental sensitivity and laboratory handling requirements.
    """

    def __init__(self):


        self.known_safe_compounds = {"NaCl", "KCl", "SiC", "TiO2", "Al2O3", "SiO2", "W", "C", "Si"}
        self.known_glovebox_compounds = {"LiH", "NaH", "CaH2", "LiAlH4", "NaBH4", "NaN3"}


        self.reactive_metals = {"Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"}
        self.toxic_metals = {"Pb", "Hg", "Cd", "As", "Tl", "Co", "Ni", "Cr", "V", "Be", "U", "Os"}
        self.halogens = {"F", "Cl", "Br", "I"}

    def evaluate(self, formula: str, state: str = "powder") -> dict:
        """
        Evaluates a chemical formula based on context, class, and state.
        state options: "bulk", "powder", "gas", "liquid"
        """

        if formula in self.known_safe_compounds:
            return {
                "material": formula,
                "classification": "Bench Safe",
                "reason": "Known stable inert compound/solid.",
                "confidence": "High"
            }
        if formula in self.known_glovebox_compounds:
            return {
                "material": formula,
                "classification": "Glovebox Required",
                "reason": "Known highly reactive or air/moisture-sensitive compound.",
                "confidence": "High"
            }


        try:

            safe_formula = re.sub(r'[^A-Za-z0-9\(\)]', '', formula)
            comp = Composition(safe_formula)
            elements = {e.symbol for e in comp.elements}
        except Exception:
            return {
                "material": formula,
                "classification": "Fume Hood Recommended",
                "reason": "Formula could not be parsed. Defaulting to safe handling.",
                "confidence": "Low"
            }


        if len(elements) == 1:
            el = list(elements)[0]


            if el in self.reactive_metals:
                return {
                    "material": formula,
                    "classification": "Glovebox Required",
                    "reason": f"Pure highly reactive metal ({el}). Reacts violently with moisture/air.",
                    "confidence": "High"
                }


            if el in self.toxic_metals:
                if state == "powder":
                    return {
                        "material": formula,
                        "classification": "Fume Hood Recommended",
                        "reason": f"Pure toxic metal ({el}). Powder poses severe inhalation hazard.",
                        "confidence": "High"
                    }
                else:
                    return {
                        "material": formula,
                        "classification": "Bench Safe",
                        "reason": f"Bulk {el} is stable. Wash hands after handling. Avoid generating dust.",
                        "confidence": "Medium"
                    }


            return {
                "material": formula,
                "classification": "Bench Safe",
                "reason": f"Stable pure element ({el}).",
                "confidence": "High"
            }


        is_oxide = "O" in elements
        is_halide = bool(elements.intersection(self.halogens))
        has_toxic = bool(elements.intersection(self.toxic_metals))
        has_reactive = bool(elements.intersection(self.reactive_metals))
        is_hydride = "H" in elements and not is_oxide  


        if is_hydride and has_reactive:
            return {
                "material": formula,
                "classification": "Glovebox Required",
                "reason": "Reactive metal hydride. Highly sensitive to moisture/air.",
                "confidence": "High"
            }


        if has_toxic:
            if state == "powder":
                return {
                    "material": formula,
                    "classification": "Fume Hood Recommended",
                    "reason": "Contains toxic transition/heavy metals. Powder state presents inhalation/exposure risk.",
                    "confidence": "High"
                }
            else:
                return {
                    "material": formula,
                    "classification": "Bench Safe",
                    "reason": "Contains toxic metals, but bulk/solid state minimizes exposure risk. Use standard PPE.",
                    "confidence": "Medium"
                }



        if is_oxide:
            return {
                "material": formula,
                "classification": "Bench Safe",
                "reason": "Stable oxide compound. Reactive elements are pacified.",
                "confidence": "High"
            }
        if is_halide:
            return {
                "material": formula,
                "classification": "Bench Safe",
                "reason": "Stable ionic halide salt. Reactive elements are pacified.",
                "confidence": "High"
            }


        return {
            "material": formula,
            "classification": "Bench Safe",
            "reason": "No high-risk functional groups, toxic metals, or free reactive elements detected.",
            "confidence": "Medium"
        }





if __name__ == "__main__":
    engine = AdvancedSafetyEngine()

    test_cases = [
        ("NaCl", "powder"),  
        ("TiO2", "powder"),  
        ("SiC", "bulk"),  
        ("W", "bulk"),  
        ("Li", "bulk"),   
        ("LiCoO2", "powder"),  
        ("LiCoO2", "bulk"),  
        ("NaBH4", "powder") 
    ]

    import pandas as pd

    results = [engine.evaluate(mat, state) for mat, state in test_cases]
    df = pd.DataFrame(results)
    print(df.to_string(index=False))