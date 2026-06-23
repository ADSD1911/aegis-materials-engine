import numpy as np
import pandas as pd
import math


class LevelAutoFixer:
    """Intelligently auto-completes missing experimental levels."""


    SMART_CATEGORIES = {
        "atmosphere": ["Ar", "N2", "Air", "Vacuum", "He", "O2"],
        "catalyst": ["Pt", "Pd", "Ni", "Ru", "Co", "Fe"],
        "solvent": ["Water", "Ethanol", "Methanol", "Acetone", "IPA", "DMF"]
    }

    @staticmethod
    def fix_levels(factor_name, provided_levels, target_count):
        levels = list(provided_levels)
        if len(levels) == target_count:
            return levels

        if len(levels) > target_count:
            return levels[:target_count]


        try:
            num_levels = [float(x) for x in levels]
            is_numeric = True
        except ValueError:
            is_numeric = False


        if is_numeric:
            if len(num_levels) >= 2:

                new_levels = np.linspace(min(num_levels), max(num_levels), target_count)
            else:

                base = num_levels[0]
                step = base * 0.1 if base != 0 else 1.0
                new_levels = [base + i * step for i in range(target_count)]


            return [int(x) if x % 1 == 0 else round(x, 4) for x in new_levels]


        new_levels = levels.copy()


        matching_key = next((k for k in LevelAutoFixer.SMART_CATEGORIES if k in factor_name.lower()), None)

        if matching_key:
            smart_list = LevelAutoFixer.SMART_CATEGORIES[matching_key]

            for item in smart_list:
                if item not in new_levels:
                    new_levels.append(item)
                if len(new_levels) == target_count:
                    break


        while len(new_levels) < target_count:
            new_levels.append(f"{factor_name}_Auto{len(new_levels) + 1}")

        return new_levels[:target_count]


class TaguchiMath:
    """Generates pure orthogonal arrays mathematically."""

    @staticmethod
    def get_array(design_type):
        arrays = {
            "L4": {"runs": 4, "factors": 3, "levels": 2, "matrix": [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]},
            "L8": {"runs": 8, "factors": 7, "levels": 2,
                   "matrix": [[bin(i & j).count('1') % 2 for j in range(1, 8)] for i in range(8)]},
            "L9": {"runs": 9, "factors": 4, "levels": 3,
                   "matrix": [[0, 0, 0, 0], [0, 1, 2, 1], [0, 2, 1, 2], [1, 0, 1, 2], [1, 1, 0, 0], [1, 2, 2, 1],
                              [2, 0, 2, 1], [2, 1, 1, 2], [2, 2, 0, 0]]},
            "L12": {"runs": 12, "factors": 11, "levels": 2,
                    "matrix": [([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0][-i:] + [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0][:-i]) for i
                               in range(11)] + [[0] * 11]},
            "L16": {"runs": 16, "factors": 15, "levels": 2,
                    "matrix": [[bin(i & j).count('1') % 2 for j in range(1, 16)] for i in range(16)]},
            "L25": {"runs": 25, "factors": 6, "levels": 5, "matrix": [
                [(i // 5) % 5, i % 5, ((i // 5) + (i % 5)) % 5, ((i // 5) + 2 * (i % 5)) % 5,
                 ((i // 5) + 3 * (i % 5)) % 5, ((i // 5) + 4 * (i % 5)) % 5] for i in range(25)]},
            "L27": {"runs": 27, "factors": 13, "levels": 3, "matrix": [
                [((i // 9) * v[0] + ((i // 3) % 3) * v[1] + (i % 3) * v[2]) % 3 for v in
                 [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2),
                  (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2)]] for i in range(27)]}
        }
        return arrays.get(design_type, None)

    @staticmethod
    def auto_select_design(num_factors, num_levels):
        """Smart suggestion engine for standard arrays."""
        if num_levels <= 2:
            if num_factors <= 3: return "L4"
            if num_factors <= 7: return "L8"
            if num_factors <= 11: return "L12"
            return "L16"
        elif num_levels == 3:
            if num_factors <= 4: return "L9"
            return "L27"
        elif num_levels <= 5:
            return "L25"
        return "L27"  # Fallback


class DoEEngine:
    """The master orchestrator."""

    def __init__(self, target_design=None, user_factors=None):
        self.factors = user_factors or {}
        num_factors = len(self.factors)
        num_levels = max([len(v) for v in self.factors.values()]) if self.factors else 2

        self.design_type = target_design or TaguchiMath.auto_select_design(num_factors, num_levels)
        self.array_config = TaguchiMath.get_array(self.design_type)

        if not self.array_config:
            raise ValueError("Invalid design type specified.")

        self.warnings = []

    def generate(self, randomize=True):
        req_factors = self.array_config['factors']
        req_levels = self.array_config['levels']
        runs = self.array_config['runs']
        base_matrix = self.array_config['matrix']


        clean_factors = {}
        provided_keys = list(self.factors.keys())

        for i in range(req_factors):

            if i < len(provided_keys):
                f_name = provided_keys[i]
                provided_vals = self.factors[f_name]
            else:
                f_name = f"Unused_Factor_{i + 1}"



            clean_vals = LevelAutoFixer.fix_levels(f_name, provided_vals, req_levels)
            clean_factors[f_name] = clean_vals

            if len(provided_vals) != req_levels and not f_name.startswith("Unused"):
                self.warnings.append(
                    f"Auto-generated levels for '{f_name}' to meet {req_levels}-level matrix requirements.")


        coded_data = []
        real_data = []

        for r in range(runs):
            coded_row = [r + 1]  
            real_row = [r + 1]

            for f_idx, (f_name, f_levels) in enumerate(clean_factors.items()):

                level_idx = base_matrix[r][f_idx]

                coded_row.append(level_idx + 1)  
                real_row.append(f_levels[level_idx])

            coded_data.append(coded_row)
            real_data.append(real_row)

        columns = ["Std_Order"] + list(clean_factors.keys())

        df_coded = pd.DataFrame(coded_data, columns=columns)
        df_real = pd.DataFrame(real_data, columns=columns)


        if randomize:
            df_real['Run_Order'] = np.random.permutation(runs) + 1
            df_coded['Run_Order'] = df_real['Run_Order']


            cols = ['Run_Order', 'Std_Order'] + list(clean_factors.keys())
            df_real = df_real[cols].sort_values('Run_Order').reset_index(drop=True)
            df_coded = df_coded[cols].sort_values('Run_Order').reset_index(drop=True)

        return {
            "metadata": {
                "design": self.design_type,
                "runs": runs,
                "factors_utilized": len([k for k in clean_factors.keys() if not k.startswith("Unused")]),
                "messages": self.warnings if self.warnings else ["Perfect design mapping achieved."]
            },
            "matrix_real": df_real,
            "matrix_coded": df_coded
        }
