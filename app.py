import streamlit as st
import pandas as pd
import py3Dmol
from pymatgen.core import Composition
from doe_engine import DoEEngine

from post_processing import apply_correction_pipeline
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import sqlite3
import streamlit.components.v1 as components
from pymatgen.core import Element
import scipy.constants as const
import joblib
import requests
import os
import re
import contextlib
import scipy.optimize as opt
import plotly.graph_objects as go
import json
import hashlib
import scipy.signal as signal
import numpy as np
import urllib.request
import math
import sympy as sp
import plotly.express as px
from datetime import datetime
from openai import OpenAI
import urllib.parse
from scipy.special import erf
from scipy.integrate import solve_ivp
from PIL import Image

import zipfile
#data 



if not os.path.exists("structure_db"):
    with st.spinner("Loading structure database — first run only, takes ~2 minutes..."):
        try:
            url = "https://github.com/ADSD1911/aegis-materials-engine/releases/download/v1.0/structure_db.zip"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=300)
            response.raise_for_status()
            with open("structure_db.zip", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile("structure_db.zip", "r") as z:
                z.extractall(".")
            os.remove("structure_db.zip")
        except Exception as e:
            st.warning(f" Structure database unavailable. 3D viewer limited this session. All other features work normally.")
# UI/UX 

try:
    aegis_logo = Image.open("materials-science.jpg")
except:
    aegis_logo = "⬡" # Fallback 

st.set_page_config(page_title="Aegis Engine", layout="wide", page_icon=aegis_logo)

#  CSS
st.markdown("""
    <style>
    /* Global Background and Fonts */
    .stApp { background-color: #0E1117; color: #E0E6ED; font-family: 'Inter', -apple-system, sans-serif; }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}


    /* Premium Tab Styling */
    .stTabs [data-baseweb="tab-list"] {gap: 8px; background-color: transparent; border-bottom: 1px solid #2D3748;}
    .stTabs [data-baseweb="tab"] {padding: 12px 28px; border-radius: 6px 6px 0px 0px; font-weight: 600; color: #A0AEC0; transition: all 0.3s ease; border: 1px solid transparent;}
    .stTabs [data-baseweb="tab"]:hover {color: #00E676; background-color: rgba(0, 230, 118, 0.05);}
    .stTabs [aria-selected="true"] {background-color: #1A202C !important; color: #00E676 !important; border: 1px solid #2D3748; border-bottom: 2px solid #00E676;}

    /* Inputs & Buttons */
    .stTextInput>div>div>input {border-radius: 6px; border: 1px solid #2D3748; background-color: #1A202C; color: #FFF;}
    .stTextInput>div>div>input:focus {border-color: #00E676; box-shadow: 0 0 0 1px #00E676;}
    .stButton>button {border-radius: 6px; font-weight: 600; border: 1px solid #2D3748; transition: all 0.2s;}
    .stButton>button:hover {border-color: #00E676; color: #00E676; box-shadow: 0 4px 12px rgba(0, 230, 118, 0.15); transform: translateY(-1px);}

    /* Primary Action Buttons */
    button[data-testid="baseButton-primary"] {background: linear-gradient(135deg, #00C853 0%, #00E676 100%); color: #000 !important; border: none;}
    button[data-testid="baseButton-primary"]:hover {box-shadow: 0 6px 20px rgba(0, 230, 118, 0.4); transform: translateY(-2px);}

    /* Premium Metric Cards */
    [data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 800; color: #00E676; text-shadow: 0 0 20px rgba(0, 230, 118, 0.2);}
    [data-testid="stMetricLabel"] {font-size: 0.85rem; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;}

    /* Containers & DataFrames */
    .stDataFrame {border-radius: 8px; border: 1px solid #2D3748;}
    [data-testid="stVerticalBlock"] {border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)



# CORE SESSION MEMORY

if "active_mat" not in st.session_state: st.session_state.active_mat = None
if "batch_data" not in st.session_state: st.session_state.batch_data = pd.DataFrame()


#  DATA 

@st.cache_resource
def load_ai_brain():
    """Loads the XGBoost core into RAM once for speed."""
    try:

        if not os.path.exists('npl_model.pkl') or not os.path.exists('npl_features.pkl'):
            st.sidebar.warning(" AI Model files (.pkl) missing from directory.")
            return None, None
            
        m = joblib.load('npl_model.pkl')
        f = joblib.load('npl_features.pkl')
        return m, f
    except Exception as e:
        st.sidebar.error(f" AI Model Load Error: {e}")
        return None, None


xgboost_model, features = load_ai_brain()


@st.cache_data
def load_vault():
    if not os.path.exists("mp_raw_data.parquet"):
        return None
    return pd.read_parquet("mp_raw_data.parquet")

df_master = load_vault()



def check_db_health(db_name):
    """Verifies if the database exists and isn't a 0-byte ghost file."""
    if not os.path.exists(db_name):
        return False, "File Missing"
    if os.path.getsize(db_name) == 0:
        return False, "Empty/Corrupt File"
    return True, "Healthy"

def get_element_dna(el_symbol):
    """The master query function. Pulls from 3 databases instantly."""
    dna = {"price": 0.0, "safety": "Standard", "scarcity": 5, "issue": "N/A"}
    
    db_configs = {
        "economics": ("db_economics.sqlite", "price"),
        "safety": ("db_safety.sqlite", "hazard_codes"),
        "sustainability": ("db_sustainability.sqlite", "scarcity_score")
    }

    for key, (db_file, col) in db_configs.items():
        is_ok, status = check_db_health(db_file)
        if not is_ok:
            
            continue
            
        try:
            with contextlib.closing(sqlite3.connect(db_file)) as conn:
                res = conn.execute(f"SELECT {col} FROM {key} WHERE element=?", (el_symbol,)).fetchone()
                if res:
                    if key == "sustainability":
                        res_ext = conn.execute("SELECT scarcity_score, issue FROM sustainability WHERE element=?", (el_symbol,)).fetchone()
                        dna["scarcity"], dna["issue"] = res_ext[0], res_ext[1]
                    else:
                        dna[key if key != "economics" else "price"] = res[0]
        except Exception as e:
            st.sidebar.error(f"SQL Error in {db_file}: {e}")
            
    return dna

def fetch_mechanical_tensor(mat_id):
    """Fetches real elasticity data using dynamic schema detection."""
    db_file = 'db_mechanics.sqlite'
    is_ok, status = check_db_health(db_file)
    if not is_ok:
        return (None, None, f"DB {status}")
        
    try:
        with contextlib.closing(sqlite3.connect(db_file)) as conn:
            
            cols_info = conn.execute("PRAGMA table_info(mechanical_props)").fetchall()
            existing_cols = [c[1] for c in cols_info]
            
            id_col = "material_id" if "material_id" in existing_cols else "mp_id" if "mp_id" in existing_cols else None
            
            if id_col:
                query = f"SELECT bulk_modulus, shear_modulus FROM mechanical_props WHERE {id_col}=?"
                res = conn.execute(query, (mat_id,)).fetchone()
                if res:
                    return (res[0], res[1], "Verified Database Match")
            else:
                st.sidebar.error(" Mechanics DB Schema Error: ID column not found.")
    except Exception as e:
        st.sidebar.error(f"SQL Error in {db_file}: {e}")
    
    return (None, None, "No record in DB")

# SIDEBAR & CONTROLS

with st.sidebar:

    logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
    with logo_col2:
        try:
            st.image("materials-science.jpg", use_container_width=True)
        except:
            pass 


    st.markdown("<h2 style='text-align: center; color: #FFF; margin-top: -15px;'> 𝑨𝑬𝑮𝑰𝑺 <span style='color:#00E676;'>𝑬𝑵𝑮𝑰𝑵𝑬</span></h2>",
                unsafe_allow_html=True)
    
    st.caption("<div style='text-align: center;'><b>Operator:</b> <code>PUBLIC VISITOR</code><br>", unsafe_allow_html=True)
    


    st.divider()


    st.markdown("### System Health")
    st.caption("Program: **Online**" if xgboost_model else "Program: **Offline**")
    st.caption("Structural Vault: **Connected**" if os.path.exists("structure_db") else "Structural Vault: **Missing**")
    st.caption(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    


    st.divider()
    st.markdown("###  Emergency Spill Guide")
    with st.expander("Rapid-Access Protocols"):
        spill = st.selectbox("Hazard / Chemical:", [
            "HF Acid (Hydrofluoric)", 
            "Piranha Solution", 
            "Alkali Metal Fire (Li, Na, K)", 
            "Mercury (Hg) Spill",
            "Strong Acids (HCl, H2SO4, HNO3)", 
            "Strong Bases (NaOH, KOH)",
            "Flammable Solvents (Acetone, Ethanol)",
            "Toxic Gas Alarm (H2S, NH3)"
        ])
        
        if "HF Acid" in spill: 
            st.error("**EVACUATE. NEUROTOXIN & BONE DESTROYER.**\nApply Calcium Gluconate gel immediately. Seek emergency medical attention.")
        elif "Piranha" in spill: 
            st.error("**EXPLOSION HAZARD.**\nDo NOT use paper towels or organics. Dilute with massive amounts of cold water.")
        elif "Alkali Metal" in spill:
            st.error("**CLASS D FIRE. DO NOT USE WATER.**\nSmother with dry sand, graphite powder, or a Class D fire extinguisher.")
        elif "Mercury" in spill:
            st.warning("**NEUROTOXIN VAPORS.**\nVentilate area. Do not use a standard vacuum. Cover with sulfur powder or commercial amalgamating agent.")
        elif "Strong Acids" in spill: 
            st.warning("**CORROSIVE.**\nNeutralize carefully with Sodium Bicarbonate (baking soda), working from the outside edges inward.")
        elif "Strong Bases" in spill:
            st.warning("**CORROSIVE.**\nNeutralize with a weak acid (e.g., 5% Acetic Acid/Vinegar or Citric Acid) before cleaning.")
        elif "Toxic Gas" in spill:
            st.error("**IMMEDIATE EVACUATION.**\nTrigger local lab alarms. Do not re-enter without SCBA gear.")
        else: 
            st.info("**FLAMMABLE.**\nTurn off all nearby hotplates and ignition sources immediately. Absorb with inert sand, vermiculite, or spill pads.")


    st.divider()
    st.markdown("### System Directory")
    with st.expander("Explore Aegis Capabilities"):
        st.markdown("""
        **0. Vault**
        - Local Material Database (143k)

        **1. Screening**
        - AI Property Prediction
        - Safety & Handling Config
        - DoE Matrix Builder

        **2. Structure**
        - 3D Lattice Viewer
        - Polymorph Dictionary
        - Live Literature Search
        - Crystallography Engine

        **3. Assistant**
        - Aegis AI Chat
        - Grant Auto-Drafter
        - Discovery Engine

        **4. Thermo**
        - Structural Density & Optics
        - Thermal Simulation
        - Fundamental Physics

        **5. Equations**
        - 18 Core Calculators
        - Advanced Computing Modules

        **6. Formats**
        - SI & Unit Converters
        - CIF / TEM / SEM Scales
        - Alloy Mass Scaling

        **7. Lab**
        - Synthesis Mass Breakdown
        - Global Economics & Scarcity
        - Crucible & Etchant Matcher

        **8. Spectra**
        - Savitzky-Golay Scrubbing
        - Auto-Peak Detection

        **9. Mechanics**
        - Elasticity & Stress-Strain
        - Hardness Converter
        - Mohr's Circle & Fracture

        **10. Elements**
        - Interactive Periodic Table
        - Atomic Radar Charts

        **11. Fields**
        - Heat & Wave PDEs
        - Lorenz Chaos & Laplace

        **12. Quantum**
        - Monte Carlo (Ising)
        - Wave Packets & Band Structure
        - Density of States

        **13. Utilities**
        - Nonlinear Curve Fitting
        - Bifurcation & Phase Maps

        **14. Devices**
        - SRH & Tunneling Kinetics
        - Drift-Diffusion & MOSFETs

        **15. Vacuum**
        - Mean Free Path & Paschen
        - Effusion & Thin-Film Stress

        **16. Solid State**
        - Reciprocal Lattices
        - Born Stability & Drude Model

        **17. Transport**
        - Gibbs Phase & Heat Transfer
        - 1D Diffusion Profiles

        **18. Math**
        - Tensor Rotations
        - ODE Solvers & Symmetry

        **19. Symbolic**
        - Equation Simplifier
        - Calculus & Auto-Derivation

        **20. Engineering**
        - Buckingham Pi Engine
        - Sensitivity Ranker
        - Goal Seek (Inverse Solver)
        """)




#  MAIN DASHBOARD

st.markdown("<h2 style='font-weight: 800;'>Dashboard</h2>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("#### Input Material")
    st.caption("Search by formula or MP-ID to load a material into the Aegis Engine.")
    
    c_search, c_btn = st.columns([5, 1])
    
    with c_search:
        search_query = st.text_input("Search:", value="", placeholder="e.g., mp-149 or Si", label_visibility="collapsed")
    with c_btn:
        search_btn = st.button(" Load ", type="primary", use_container_width=True)

    if search_btn:
        if search_query:
            with st.spinner("Querying Vault..."):
                try:
                    if df_master is None:
                        st.error(" Core Index Missing: 'mp_raw_data.parquet' not found.")
                    else:
                        query_list = [q.strip() for q in search_query.split(",") if q.strip()]
                        targets = df_master[(df_master['material_id'].isin(query_list)) | (df_master['formula'].isin(query_list))].copy()

                        if not targets.empty:

                            if 'formation_energy_per_atom' in targets.columns:
                                targets['id_len'] = targets['material_id'].astype(str).str.len()
                                targets = targets.sort_values(by=['formation_energy_per_atom', 'id_len'], ascending=[True, True])
                                targets = targets.drop(columns=['id_len'])


                            targets = targets.drop_duplicates(subset=['formula'], keep='first')



                            st.session_state.active_mat = targets.iloc[0].to_dict()
                            
                            target_formula = targets.iloc[0].get('formula', 'Unknown')
                            st.success(f" Locked onto standard phase for {target_formula}.")
                        else:
                            st.error("No matches found in the 143k local index.")
                except Exception as e:
                    st.error(f" Search Engine Error: {e}")
        else:
            st.warning("Please enter a material ID or formula.")
if st.session_state.active_mat:
    st.info(
        f"**Active Target:** {st.session_state.active_mat['formula']} ({st.session_state.active_mat['material_id']})")
st.divider()



tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20 = st.tabs([
    "Vault", 
    "Screening", 
    "Structure", 
    "Assistant", 
    "Thermo",
    "Equations", 
    "Formats", 
    "Lab", 
    "Spectra", 
    "Mechanics", 
    "Elements", 
    "Fields", 
    "Quantum", 
    "Utilities",
    "Devices",
    "Vacuum",
    "Solid State",
    "Transport",
    "Math",
    "Symbolic",
    "Engineering"
])

#  TAB 0 



with tab0:
    st.markdown("### Local Vault Explorer <span style='color:#F56565; font-size:0.5em; vertical-align:middle; border:1px solid #F56565; padding:2px 6px; border-radius:4px; margin-left:8px;'>BETA</span>", unsafe_allow_html=True)
    st.caption("Query your local 143k database to find the exact global MP-ID for any structural polymorph.")

    c1, c2 = st.columns([1, 2])
    with c1:
        ref_search = st.text_input("Find Polymorphs (Enter Formula):", placeholder="e.g., TiO2, Fe, C").strip()
    
    with c2:
        st.write("")
        st.write("")
        st.caption("* Results are sorted by thermodynamic stability (Energy Above Hull) to prioritize ground states.*")

    if ref_search:
        if df_master is None:
            st.error(" Core Index Missing: 'mp_raw_data.parquet' not found in the root directory.")
        else:
            with st.spinner(f"Scanning Vault for '{ref_search}'..."):
                try:
                    matches = df_master[df_master['formula'] == ref_search].copy()
                    
                    if not matches.empty:

                        sort_col = next((col for col in ['e_above_hull', 'formation_energy_per_atom', 'energy_per_atom'] if col in matches.columns), None)
                        
                        if sort_col:
                            matches['id_len'] = matches['material_id'].astype(str).str.len()
                            matches = matches.sort_values(by=[sort_col, 'id_len'], ascending=[True, True])

                        display_data = {
                            "MP-ID (Copy this)": matches['material_id'],
                            "Formula": matches['formula'],
                        }
                        

                        def clean_crystal_system(cs):
                            if pd.isna(cs): return "Unknown"
                            cs_str = str(cs).lower()
                            if "hex" in cs_str: return "Hexagonal"
                            if "tet" in cs_str: return "Tetragonal"
                            if "tri" in cs_str: return "Triclinic"
                            if "ortho" in cs_str: return "Orthorhombic"
                            if "mono" in cs_str: return "Monoclinic"
                            if "cub" in cs_str: return "Cubic"
                            if "rhom" in cs_str or "trig" in cs_str: return "Rhombohedral"
                            return str(cs).title()

                        cs_col = next((col for col in ['crystal_system', 'symmetry.crystal_system', 'symmetry__crystal_system'] if col in matches.columns), None)
                        if cs_col:
                            display_data["Crystal System"] = matches[cs_col].apply(clean_crystal_system)
                            

                        sg_col = next((col for col in ['symmetry.symbol', 'spacegroup.symbol', 'spacegroup_symbol', 'spacegroup'] if col in matches.columns), None)
                        if sg_col:
                            display_data["Space Group"] = matches[sg_col]
                            

                        if 'e_above_hull' in matches.columns:

                            display_data["Stability (e_above_hull)"] = matches['e_above_hull'].apply(
                                lambda x: f"{x:.3f} eV (Stable 🟢)" if float(x) <= 0.005 else f"{x:.3f} eV"
                            )
                        elif 'formation_energy_per_atom' in matches.columns:
                            display_data["Formation Energy (eV/atom)"] = matches['formation_energy_per_atom'].apply(lambda x: f"{x:.4f}")


                        if 'band_gap' in matches.columns:
                            display_data["Band Gap (eV)"] = matches['band_gap'].apply(lambda x: f"{x:.2f}")
                            
                        mag_col = next((col for col in ['ordering', 'is_magnetic', 'magnetic_type'] if col in matches.columns), None)
                        if mag_col:
                            display_data["Magnetism"] = matches[mag_col].replace({True: "Yes", False: "No", "NM": "Non-Magnetic", "FM": "Ferromagnetic", "AFM": "Anti-Ferromagnetic"})
                            
                        if 'density' in matches.columns:
                            display_data["Density (g/cm³)"] = matches['density'].apply(lambda x: f"{x:.2f}")


                        display_df = pd.DataFrame(display_data)
                        
                        st.success(f" Found {len(matches)} known polymorphs for **{ref_search}**.")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                    else:
                        st.warning(f" No materials found matching '{ref_search}'. Ensure proper chemical capitalization (e.g., use 'TiO2' instead of 'tio2').")
                except Exception as e:
                    st.error(f" Database Scan Error: {e}")
    else:
        st.info(" Enter a chemical formula above to view all its structural variations and extract their specific MP-IDs.")

#  TAB 1 

with tab1:
    
    st.markdown("### Material Screening")
    st.caption("Deploy the XGBoost model to predict formation energy and electronic state from chemical formulas.")

    batch_input = st.text_input("Target Materials (Comma Separated):", value="",
                                placeholder="e.g., TiO2, NaCl, W, SiC, LiCoO2")

    if st.button("Run ", type="primary"):

        if not xgboost_model:
            st.error(" AI Offline: 'npl_model.pkl' or 'npl_features.pkl' missing or corrupted.")
        elif not batch_input:
            st.warning("Please enter at least one material formula.")
        else:
            with st.spinner("Featurizing and executing AI inference..."):
                try:
                    formulas = [f.strip() for f in batch_input.split(",") if f.strip()]
                    df_in = pd.DataFrame({'formula': formulas})


                    df_in = StrToComposition().featurize_dataframe(df_in, "formula", ignore_errors=True)
                    df_in = df_in.dropna(subset=['composition'])
                    ep = ElementProperty.from_preset("magpie")
                    df_in = ep.featurize_dataframe(df_in, col_id="composition", ignore_errors=True)





                    X_raw = df_in.reindex(columns=features, fill_value=0).fillna(0)
                    preds = xgboost_model.predict(X_raw)


                    corrected_energies = []
                    corrected_states = []
                    corrected_confidences = []


                    for idx, formula in enumerate(df_in['formula']):
                        if preds.ndim == 2 and preds.shape[1] >= 2:
                            raw_energy = float(preds[idx, 0])
                            raw_state = "Metal" if preds[idx, 1] > 0.5 else "Insulator"
                        elif preds.ndim == 1:
                            raw_energy = float(preds[idx])
                            raw_state = "Unknown" 
                        else:
                            raise ValueError(f"Unexpected prediction shape: {preds.shape}")


                        corrected = apply_correction_pipeline(
                            formula=formula,
                            predicted_energy=raw_energy,
                            predicted_state=raw_state
                        )


                        corrected_energies.append(corrected["energy_ev_atom"])
                        corrected_states.append(corrected["electronic_state"])
                        corrected_confidences.append(corrected["confidence"])


                    results_df = pd.DataFrame({
                        "Formula": df_in['formula'],
                        "Formation Energy (eV/atom)": corrected_energies,
                        "Electronic State": corrected_states,
                        "Model Confidence": corrected_confidences
                    })

                    st.session_state.batch_data = results_df
                    st.success(f" Processed & Corrected {len(results_df)} materials.")
                except Exception as e:
                    st.error(f" Inference Error: Check chemical formula formatting. ({e})")

    if not st.session_state.batch_data.empty:
        styled_df = st.session_state.batch_data.style.background_gradient(subset=['Formation Energy (eV/atom)'],
                                                                          cmap='RdYlGn_r')
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


# ENVIRONMENTAL SAFETY
    st.divider()
    st.markdown("### Environmental Sensitivity & Handling")
    st.caption(" **Notice:** This engine is calibrated for solid-state materials. Accuracy is not guaranteed for general liquid/gas chemistry like $NH_3$ or $H_2SO_4$.")

    if not st.session_state.batch_data.empty:
        m_state = st.radio("Expected Material State:", ["Powder", "Bulk"], horizontal=True)

        with st.spinner("Analyzing Chemical Context..."):
            try:
                from engines.advanced_safety_engine import AdvancedSafetyEngine
                safe_eng = AdvancedSafetyEngine()
            except ModuleNotFoundError:
                st.error(" Safety Engine missing: 'engines/advanced_safety_engine.py' not found.")
                st.stop()

            gb, fh, safe = [], [], []
            for f in st.session_state.batch_data["Formula"]:
                

                r = safe_eng.evaluate(f, state=m_state.lower())
                txt = f"**{r.get('material', f)}**: {r.get('reason', 'No data')}"
                
                classification_lower = str(r.get("classification", "")).lower()
                reason_lower = str(r.get("reason", "")).lower()


                extreme_hazards = ['Po', 'U', 'Th', 'Pu', 'Ra', 'Hg', 'Cd', 'As', 'Pb']
                has_extreme = any(hazard in f for hazard in extreme_hazards)

                if "glovebox" in classification_lower:
                    gb.append(txt)
                elif "fume hood" in classification_lower or "toxic" in reason_lower or "hazard" in reason_lower or has_extreme:

                    if has_extreme and "fume hood" not in classification_lower:
                        txt += " *(UI Safety Net: Contains extreme hazardous/radioactive elements)*"
                    fh.append(txt)
                else:
                    safe.append(txt)

        if gb:
            st.error(" **Glovebox Required**", icon="🔴")
            for i in gb: st.markdown(f"- {i}")
        if fh:
            st.warning(" **Fume Hood Recommended**", icon="🟠")
            for i in fh: st.markdown(f"- {i}")
        if safe:
            st.success(" **Safe**", icon="🟢")
            for i in safe: st.markdown(f"- {i}")
    else:
        st.info(" Run Material screening above to unlock safety analysis.")


# DOE matrix builder 

    st.divider()
    st.markdown("### Design of Experiments (DoE) Matrix Builder")
    st.caption("Define your factors dynamically in the fields below, then generate the orthogonal matrix.")

    with st.expander("Configure Taguchi Orthogonal Arrays", expanded=True):
        doe_type = st.selectbox("Select Taguchi Orthogonal Array:", [
            "L4 (3 Factors, 2 Levels)",
            "L8 (7 Factors, 2 Levels)",
            "L9 (4 Factors, 3 Levels)",
            "L12 (11 Factors, 2 Levels) - Plackett-Burman",
            "L16 (15 Factors, 2 Levels)",
            "L25 (6 Factors, 5 Levels)",
            "L27 (13 Factors, 3 Levels)"
        ])

        factors = int(re.search(r'\((\d+) Factors', doe_type).group(1))
        levels = int(re.search(r'(\d+) Levels\)', doe_type).group(1))

        default_names = ["Temperature (°C)", "Pressure (atm)", "Time (hrs)", "Ramp Rate (°C/min)", "Atmosphere",
                         "Catalyst (wt%)", "pH", "Stir Rate (rpm)", "Solvent Ratio"]
        default_l1 = ["800", "1.0", "2.0", "5", "Ar", "1.0", "6.0", "100", "1:1"]
        default_l2 = ["1000", "5.0", "12.0", "10", "N2", "3.0", "7.0", "300", "1:2"]
        default_l3 = ["1200", "10.0", "24.0", "20", "Air", "5.0", "8.0", "500", "1:3"]

        st.markdown("**1. Define your Factors & Levels (Type directly into the boxes):**")

        factor_configs = []

        for i in range(factors):
            cols = st.columns(levels + 1)
            label_vis = "visible" if i == 0 else "collapsed"

            def_name = default_names[i] if i < len(default_names) else f"Factor {i + 1}"
            fname = cols[0].text_input("Factor Name", value=def_name, key=f"fname_{i}", label_visibility=label_vis)

            vals = []
            for l in range(levels):
                if l == 0:
                    def_val = default_l1[i] if i < len(default_l1) else "Low"
                elif l == 1:
                    def_val = default_l2[i] if i < len(default_l2) else "High"
                elif l == 2:
                    def_val = default_l3[i] if i < len(default_l3) else "Max"
                else:
                    def_val = f"Val {l + 1}"

                v = cols[l + 1].text_input(f"Level {l + 1}", value=def_val, key=f"f{i}_l{l}",
                                           label_visibility=label_vis)
                vals.append(v)

            factor_configs.append({"name": fname, "vals": vals})

        if st.button("Generate DoE Matrix", type="primary"):
            with st.spinner("Engine calculating orthogonal arrays..."):

                user_inputs = {}
                for f_idx in range(factors):
                    f_name = factor_configs[f_idx]["name"]
                    raw_vals = [v for v in factor_configs[f_idx]["vals"] if str(v).strip() != ""]

                    clean_vals = []
                    for v in raw_vals:
                        try:
                            clean_vals.append(float(v))
                        except ValueError:
                            clean_vals.append(v)

                    if clean_vals:
                        user_inputs[f_name] = clean_vals

                try:
                    target_l = doe_type.split(" ")[0]

                    engine = DoEEngine(target_design=target_l, user_factors=user_inputs)
                    payload = engine.generate(randomize=True)

                    st.success(f" {target_l} Matrix Generated Successfully!")

                    for msg in payload["metadata"]["messages"]:
                        if "Auto-generated" in msg:
                            st.caption(f"  System Note: {msg}")

                    st.dataframe(payload["matrix_real"], hide_index=True, use_container_width=True)

                    csv = payload["matrix_real"].to_csv(index=False).encode('utf-8')
                    st.download_button("  Download Matrix as CSV", data=csv, file_name=f"DoE_{target_l}.csv",
                                       mime='text/csv')

                except Exception as e:
                    st.error(f" Engine Error: {e}")




#  TAB 2 

with tab2:
    st.markdown("### 3D Structural Engine")
    st.caption(" Lattice rendering and high-tolerance crystallographic analysis.")


    st.markdown("""
    <style>
        .c-card { background: #1A202C; border: 1px solid #2D3748; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
        .c-label { color: #A0AEC0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
        .c-value { color: #00E676; font-size: 18px; font-weight: 800; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.active_mat:
        mat = st.session_state.active_mat
        cif_path = os.path.join("structure_db", f"{mat['material_id']}.cif")

        if os.path.exists(cif_path):

            from pymatgen.core.structure import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.io.cif import CifWriter
            
            try:

                raw_struct = Structure.from_file(cif_path)
                sga = SpacegroupAnalyzer(raw_struct, symprec=0.1, angle_tolerance=5.0) 
                

                conv_struct = sga.get_conventional_standard_structure()
                

                cif_str = CifWriter(conv_struct).__str__()
                
                c_system = sga.get_crystal_system().title()
                sg_symbol = sga.get_space_group_symbol()
                sg_num = sga.get_space_group_number()
                
                bravais = "Primitive (P)"
                if "F" in sg_symbol.split()[0]: bravais = "Face-Centered (FCC)"
                elif "I" in sg_symbol.split()[0]: bravais = "Body-Centered (BCC)"
                elif "C" in sg_symbol.split()[0] or "A" in sg_symbol.split()[0]: bravais = "Base-Centered"
            except Exception as e:

                with open(cif_path, 'r') as f: cif_str = f.read()
                c_system, sg_symbol, sg_num, bravais = "Unknown", "Unknown", "N/A", "Unknown"


            st.markdown("#####  Render Settings")
            r1, r2, r3, r4 = st.columns([1.5, 1.5, 1, 1])
            style_mode = r1.selectbox("Render Style:", ["Textbook Lattice", "Thick Bonds", "Wireframe Skeleton"], label_visibility="collapsed")

            supercell = r2.slider("Lattice Multiplier:", 1, 4, 2, label_visibility="collapsed")
            auto_spin = r3.toggle(" Auto-Spin", value=True)
            bg_color = r4.color_picker("Background", value="#0E1117", label_visibility="collapsed") 


            st.markdown(f"<h2 style='color:#00E676; margin-bottom:-15px;'>{mat.get('formula', 'Unknown')}</h2>", unsafe_allow_html=True)
            st.caption(f"**Target ID:** {mat.get('material_id', 'Unknown')} | **Geometry:** Conventional Standard Cell")
            
            with st.container(border=True):
                view = py3Dmol.view(width=800, height=450)
                view.addModel(cif_str, 'cif', {'doAssembly': True, 'duplicate': [supercell, supercell, supercell]})
                
                if style_mode == "Textbook Lattice":
                    view.setStyle({'sphere': {'radius': 0.35}, 'stick': {'hidden': True}})
                elif style_mode == "Thick Bonds":
                    view.setStyle({'sphere': {'radius': 0.45}, 'stick': {'radius': 0.25, 'colorscheme': 'Jmol'}})
                elif style_mode == "Wireframe Skeleton":
                    view.setStyle({'line': {'linewidth': 3, 'colorscheme': 'Jmol'}}) 
                    
                view.addUnitCell() 
                view.setBackgroundColor(bg_color) 
                
                if auto_spin: view.spin("y", 0.5) 
                view.zoomTo()
                components.html(view._make_html(), height=450, width=800)


            st.write("") # Spacer
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='c-card'><div class='c-label'>Crystal System</div><div class='c-value'>{c_system}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='c-card'><div class='c-label'>Bravais Lattice</div><div class='c-value'>{bravais}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='c-card'><div class='c-label'>Space Group</div><div class='c-value'>{sg_symbol}</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='c-card'><div class='c-label'>SG Number</div><div class='c-value'>No. {sg_num}</div></div>", unsafe_allow_html=True)


            st.divider()
            st.markdown("##### Thermodynamic Provenance & State")
            


            try:
                id_num = int(re.findall(r'\d+', mat['material_id'])[0])


                if id_num < 100000:
                    phase_state = "Standard Ground State (1 atm, 298K)"
                    stability = "Highly Stable (Textbook Phase)"
                    state_color = "#00E676" 
                else:
                    phase_state = "High-Pressure / Computational Allotrope"
                    stability = "Metastable (Energy > Hull)"
                    state_color = "#F56565" 
            except:
                phase_state = "Unknown Provenance"
                stability = "Unknown"
                state_color = "#A0AEC0"

            p1, p2 = st.columns(2)
            with p1.container(border=True):
                st.caption("Predicted Phase State")
                st.markdown(f"<h4 style='color:{state_color}; margin-top:-5px;'>{phase_state}</h4>", unsafe_allow_html=True)
            with p2.container(border=True):
                st.caption("Thermodynamic Stability")
                st.markdown(f"<h4 style='color:{state_color}; margin-top:-5px;'>{stability}</h4>", unsafe_allow_html=True)
        else:
            st.error(f"Missing Data: CIF file for {mat['material_id']} not found in 'structure_db'.")
    else:
        st.info("Use the Universal Vault Search on the dashboard to Unlock Material .")


    st.divider()
    st.markdown("### Known Polymorphs & Allotropes")
    allotrope_db = {"C": ["Diamond", "Graphite", "Graphene", "Fullerene"], "TiO2": ["Rutile", "Anatase", "Brookite"],
                    "Al2O3": ["Alpha", "Gamma", "Theta"], "Fe": ["Alpha-Fe", "Gamma-Fe", "Delta-Fe"],
                    "SiC": ["3C-SiC", "4H-SiC", "6H-SiC"]}


    formula_list = st.session_state.batch_data["Formula"].tolist() if not st.session_state.batch_data.empty else ["C",
                                                                                                                  "TiO2",
                                                                                                                  "Fe",
                                                                                                                  "SiC",
                                                                                                                  "Al2O3"]

    allo_data = [{"Formula": f, "Known Phases": ", ".join(allotrope_db.get(f, ["No variations in base subset"]))}
                 for f in formula_list]
    st.dataframe(pd.DataFrame(allo_data), hide_index=True, use_container_width=True)


    st.divider()
    st.markdown("### Live Literature Auto-Searcher")
    


    target_search = st.text_input(
        " Enter Material :", 
        value="Graphene", 
        key="lit_search_input",
        help="Enter any chemical formula or material name to search global literature."
    )

    if st.button(f"Fetch Latest Papers for {target_search}", key="lit_search_btn"):
        with st.spinner("Querying Crossref API..."):
            try:
                

                safe_query = urllib.parse.quote(f"{target_search} material synthesis")
                url = f"https://api.crossref.org/works?query={safe_query}&select=title,URL&rows=3"
                
                headers = {'User-Agent': 'MaterialsIntelligenceSuite/1.0 (mailto:researcher@materialssuite.com)'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                if items:
                    for paper in items:
                        title = paper.get('title', ['Unknown Title'])[0]
                        url = paper.get('URL', '#')
                        st.markdown(f"- **[{title}]({url})**")
                else:
                    st.warning(f"No recent papers found for '{target_search}'. Try a broader search.")
                    
            except requests.exceptions.Timeout:
                st.error(" Connection Timeout: The Crossref database took too long to respond (10s limit).")
            except Exception as e:
                st.error(f" API Connection Error: {e}")


    st.divider()
    st.markdown("###  Real XRD Diffraction (Cu K-α)")
    if st.session_state.active_mat:
        if st.button("Calculate Exact XRD from CIF", key="xrd_calc_btn"):
            with st.spinner("Calculating Bragg reflections from atomic coordinates..."):
                try:
                    from pymatgen.analysis.diffraction.xrd import XRDCalculator
                    from pymatgen.core.structure import Structure

                    cif_path = os.path.join("structure_db", f"{st.session_state.active_mat['material_id']}.cif")
                    

                    if not os.path.exists(cif_path):
                        st.error(f" Missing Data: CIF file not found at '{cif_path}'. Check your structure_db folder.")
                    else:
                        struct = Structure.from_file(cif_path)
                        xrd_calc = XRDCalculator(wavelength="CuKa")
                        pattern = xrd_calc.get_pattern(struct)

                        xrd_df = pd.DataFrame({"2-Theta (Degrees)": pattern.x, "Intensity (a.u.)": pattern.y})
                        fig_xrd = px.bar(xrd_df, x="2-Theta (Degrees)", y="Intensity (a.u.)",
                                         title=f"Real XRD Pattern: {st.session_state.active_mat['formula']}")
                        fig_xrd.update_traces(marker_color='#00E676', width=0.3)
                        fig_xrd.update_layout(yaxis_range=[0, 110], xaxis_range=[10, 90])
                        st.plotly_chart(fig_xrd, use_container_width=True)
                        
                except Exception as e:

                    st.error(f" XRD Calculation Error: {e}")
    else:
        st.info(" Load a material first to calculate its exact XRD.")




    st.divider()
    st.markdown("#### Advanced Crystallography Engine")
    st.caption("Contextual tools for lattice planes, diffraction limits, and surface energies.")

    cryst_adv_tool = st.selectbox("Select Crystallography Module:", [
        "1. Lattice Plane Indexing (d-spacing calculator)",
        "2. Structure Factor Rules (FCC/BCC/SC)",
        "3. Debye-Scherrer Crystallite Size Estimator"
    ])

    with st.container(border=True):
        if "Plane Indexing" in cryst_adv_tool:
            st.markdown("**Calculate interplanar d-spacing for all 7 Crystal Systems.**")
            
            c1, c2 = st.columns([1, 2.5])
            miller_hkl = c1.text_input("Miller Indices (h k l):", value="1 1 1")
            lattice_sys = c2.selectbox("Crystal System:", [
                "Cubic (a=b=c, α=β=γ=90°)", 
                "Tetragonal (a=b≠c, α=β=γ=90°)", 
                "Orthorhombic (a≠b≠c, α=β=γ=90°)",
                "Hexagonal (a=b≠c, α=β=90°, γ=120°)",
                "Monoclinic (a≠b≠c, α=γ=90°, β≠90°)",
                "Rhombohedral (a=b=c, α=β=γ≠90°)",
                "Triclinic (a≠b≠c, α≠β≠γ)"
            ])
            

            dis_b = "Cubic" in lattice_sys or "Tetragonal" in lattice_sys or "Hexagonal" in lattice_sys or "Rhombohedral" in lattice_sys
            dis_c = "Cubic" in lattice_sys or "Rhombohedral" in lattice_sys
            
            dis_alpha = not ("Triclinic" in lattice_sys or "Rhombohedral" in lattice_sys)

            dis_beta = not ("Monoclinic" in lattice_sys or "Triclinic" in lattice_sys)
            dis_gamma = not ("Triclinic" in lattice_sys)

            st.caption("Lattice Parameters (Å):")
            c3, c4, c5 = st.columns(3)
            a_lat = c3.number_input("a (Å):", value=5.0)
            b_lat = c4.number_input("b (Å):", value=5.0 if dis_b else 6.0, disabled=dis_b)
            c_lat = c5.number_input("c (Å):", value=5.0 if dis_c else 7.0, disabled=dis_c)
            
            st.caption("Lattice Angles (Degrees):")
            c6, c7, c8 = st.columns(3)
            alpha_deg = c6.number_input("α (°):", value=60.0 if "Rhombohedral" in lattice_sys else 90.0, disabled=dis_alpha)
            beta_deg = c7.number_input("β (°):", value=100.0 if "Monoclinic" in lattice_sys else (60.0 if "Rhombohedral" in lattice_sys else 90.0), disabled=dis_beta)
            gamma_deg = c8.number_input("γ (°):", value=120.0 if "Hexagonal" in lattice_sys else (60.0 if "Rhombohedral" in lattice_sys else 90.0), disabled=dis_gamma)
            
            if st.button("Calculate d-spacing", key="calc_d_tab2"):
                try:

                    clean_hkl = miller_hkl.replace(',', ' ').split()
                    h, k, l = map(int, clean_hkl)
                    

                    if "Cubic" in lattice_sys: 
                        b_lat, c_lat = a_lat, a_lat
                        al = be = ga = math.radians(90)
                    elif "Tetragonal" in lattice_sys: 
                        b_lat = a_lat
                        al = be = ga = math.radians(90)
                    elif "Orthorhombic" in lattice_sys:
                        al = be = ga = math.radians(90)
                    elif "Hexagonal" in lattice_sys:
                        b_lat = a_lat
                        al, be, ga = math.radians(90), math.radians(90), math.radians(120)
                    elif "Monoclinic" in lattice_sys:
                        al, ga, be = math.radians(90), math.radians(90), math.radians(beta_deg)
                    elif "Rhombohedral" in lattice_sys:
                        b_lat, c_lat = a_lat, a_lat
                        al = be = ga = math.radians(alpha_deg) 
                    elif "Triclinic" in lattice_sys:
                        al, be, ga = math.radians(alpha_deg), math.radians(beta_deg), math.radians(gamma_deg)
                    
                    a, b, c = a_lat, b_lat, c_lat
                    

                    V2 = (a**2 * b**2 * c**2) * (1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2 + 2 * math.cos(al) * math.cos(be) * math.cos(ga))
                    
                    if V2 <= 0:
                        st.error("Math Domain Error: Invalid lattice angles! Volume is zero or imaginary.")
                    else:
                        S11 = (b * c * math.sin(al))**2
                        S22 = (a * c * math.sin(be))**2
                        S33 = (a * b * math.sin(ga))**2
                        S12 = a * b * c**2 * (math.cos(al) * math.cos(be) - math.cos(ga))
                        S23 = a**2 * b * c * (math.cos(be) * math.cos(ga) - math.cos(al))
                        S13 = a * b**2 * c * (math.cos(ga) * math.cos(al) - math.cos(be))
                        
                        inv_d2 = (S11*h**2 + S22*k**2 + S33*l**2 + 2*S12*h*k + 2*S23*k*l + 2*S13*h*l) / V2
                        
                        if inv_d2 <= 0:
                            st.error("Invalid Miller Indices.")
                        else:
                            d_spacing = math.sqrt(1 / inv_d2)
                            st.success(f"**Interplanar Spacing ($d_{{{h}{k}{l}}}$):** {d_spacing:.4f} Å")
                except Exception as e:
                    st.error("Please enter valid space-separated integers for h, k, and l (e.g., '1 1 1' or '1, 1, 1').")
    

        elif "Structure Factor" in cryst_adv_tool:
            st.markdown("**Identify allowed vs. forbidden XRD reflections based on lattice centering.**")
            hkl_input = st.text_input("Enter Miller Indices (h k l) to check:", value="2 0 0")
            
            if st.button("Check Reflection Rules", key="btn_struct_factor_tab2"):
                try:
                    h, k, l = map(int, hkl_input.split())
                    sum_hkl = h + k + l
                    all_even_or_odd = (h%2 == k%2 == l%2)
                    
                    c1, c2, c3 = st.columns(3)
                    

                    c1.metric("Simple Cubic (SC)", "Allowed", "F = f")
                    

                    if sum_hkl % 2 == 0:
                        c2.metric("Body-Centered (BCC)", "Allowed", "F = 2f")
                    else:
                        c2.metric("Body-Centered (BCC)", "Forbidden", delta="Sum is odd", delta_color="inverse")
                        

                    if all_even_or_odd:
                        c3.metric("Face-Centered (FCC)", "Allowed", "F = 4f")
                    else:
                        c3.metric("Face-Centered (FCC)", "Forbidden", delta="Mixed parity", delta_color="inverse")
                except:
                    st.error("Please enter valid space-separated integers.")

        elif "Debye-Scherrer" in cryst_adv_tool:
            st.latex(r"\tau = \frac{K\lambda}{\beta \cos(\theta)}")
            st.markdown("**Estimate nanoparticle/crystallite size from XRD peak broadening.**")
            
            c1, c2, c3 = st.columns(3)
            shape_K = c1.number_input("Shape Factor (K):", value=0.90, min_value=0.10, help="0.9 is standard for spherical particles.")
            wave_lam = c2.number_input("Wavelength $\lambda$ (Å):", value=1.5406, min_value=0.10, help="Cu K-alpha")
            theta_deg = c3.number_input("Bragg Angle $\theta$ (Degrees):", value=20.0, min_value=0.1, max_value=89.9)
            
            c4, c5 = st.columns(2)
            fwhm_deg = c4.number_input("Peak FWHM (Degrees):", value=0.20, min_value=0.001)
            inst_broad = c5.number_input("Instrumental Broadening (Degrees):", value=0.05, min_value=0.0)
            
            if st.button("Calculate Crystallite Size ($\tau$)", key="btn_scherrer_tab2"):
                

                if fwhm_deg <= inst_broad:
                    st.error("Error: Peak FWHM must be strictly greater than Instrumental Broadening.")
                else:

                    true_broad_deg = math.sqrt(fwhm_deg**2 - inst_broad**2)
                    
                    beta_rad = math.radians(true_broad_deg)
                    theta_rad = math.radians(theta_deg)
                    

                    tau_nm = ((shape_K * wave_lam) / (beta_rad * math.cos(theta_rad))) / 10.0
                    
                    st.success(f"**Estimated Crystallite Size ($\tau$):** {tau_nm:.2f} nm")
                    
                    if tau_nm > 100:
                        st.warning(" Warning: Scherrer equation is generally unreliable for crystallites larger than ~100 nm.")

#  TAB 3 

with tab3:
    st.markdown("### Materials AI Assistant")
    st.caption("AI Assistant | Powered by Groq (Llama 3)")


    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if "active_groq_model" not in st.session_state:
        st.session_state.active_groq_model = "llama-3.3-70b-versatile"
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []


    def run_agent_query(prompt_type, user_input, model_name):
        """Unified Groq engine using the OpenAI SDK format."""
        if not st.session_state.groq_api_key:
            st.error(" API Key required in the 'Intelligence Config' section on the right.")
            return None
        
        try:
            

            client = OpenAI(
                api_key=st.session_state.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            

            system_prompt = "You are a Materials Scientist at a Laboratory. Be technical, precise, and use metric/SI units (GPa, eV, Kelvin)."
            
            context = ""
            if st.session_state.active_mat:
                m = st.session_state.active_mat
                context += f"ACTIVE_TARGET: {m.get('formula', 'Unknown')} (ID: {m.get('material_id', 'Unknown')})\n"
            
            if not st.session_state.batch_data.empty:
                valid_cols = [c for c in ['Formula', 'Formation Energy (eV/atom)'] if c in st.session_state.batch_data.columns]
                if valid_cols:
                    preview = st.session_state.batch_data[valid_cols].head(3).to_string(index=False)
                    context += f"LOCAL_BATCH_CONTEXT:\n{preview}\n"
            

            prompts = {
                "chat": f"{context}\nQUERY: {user_input}\nRESPONSE:",
                "planner": f"{context}\nGOAL: {user_input}\nTASK: Suggest specific materials, synthesis routes, and a Taguchi DoE matrix.",
                "diagnostic": f"{context}\nOBSERVATION: {user_input}\nTASK: Diagnose probable chemical/mechanical causes and suggest fixes.",
                "grant": f"{context}\nFOCUS: {user_input}\nTASK: Write a Title, NSF Abstract, Innovation statement, and Broader Impacts.",
                "critique": f"{context}\nPROPOSAL: {user_input}\nTASK: Act as a harsh peer reviewer. Identify weaknesses and score novelty (1-10).",
                "discovery": f"{context}\nTARGET: {user_input}\nTASK: Suggest 3 specific candidates ranked by thermodynamic stability and cost."
            }
            


            msg_list = [{"role": "system", "content": system_prompt}]
            
            if prompt_type == "chat":

                for h in st.session_state.ai_chat_history[:-1]:
                    msg_list.append({"role": h["role"], "content": h["content"]})
                

                msg_list.append({"role": "user", "content": prompts["chat"]})
            else:

                msg_list.append({"role": "user", "content": prompts.get(prompt_type, prompts["chat"])})

            response = client.chat.completions.create(
                model=model_name,
                messages=msg_list,
                temperature=0.2
            )
            return response.choices[0].message.content

        except Exception as e:
            err_msg = str(e).lower()
            if "authentication" in err_msg or "api key" in err_msg:
                st.error(" **Invalid API Key:** Please verify your Groq API key (starts with 'gsk_').")
            else:
                st.error(f" Groq Engine Error: {e}")
            return None


    left_col, right_col = st.columns([1.3, 1])

    # RIGHT COLUMN: CONFIGURATION
    with right_col:
        with st.container(border=True):
            st.markdown("##### Intelligence Config")

            st.selectbox("Groq Model:", 
                         ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
                         key="active_groq_model",
                         help="Llama 3.3 70B is highly recommended for complex scientific reasoning.")

        with st.container(border=True):
            st.markdown("##### Grant Auto-Drafter (NSF/DoE)")
            target_f = st.session_state.active_mat['formula'] if st.session_state.active_mat else "Current Batch"
            

            grant_topic = st.text_area("Research Objective / Proposal Idea:", placeholder=f"e.g., Using {target_f} to improve solid-state battery charge cycles...")
            
            if st.button(f"Draft Proposal for {target_f}", use_container_width=True, key="btn_grant_draft"):
                if not grant_topic.strip():
                    st.warning("Please enter a research objective first.")
                else:

                    combined_focus = f"Material: {target_f}. Objective: {grant_topic}"
                    res = run_agent_query("grant", combined_focus, st.session_state.active_groq_model)
                    
                    if res:
                        st.session_state.last_grant = res
                        st.success("Draft Generated")
                        st.markdown(res)

        with st.expander(" Reviewer Critique Mode"):
            crit_text = st.text_area("Proposal Text:", value=st.session_state.get("last_grant", ""), height=150)
            if st.button("Execute Critique", use_container_width=True):
                res = run_agent_query("critique", crit_text, st.session_state.active_groq_model)
                if res: st.warning(res)

        with st.container(border=True):
            st.markdown("##### Directed Material Discovery")
            disc_goal = st.text_input("Attributes:", placeholder="Cheap, Bandgap > 2eV, Stable in air")
            if st.button("Initiate Discovery", use_container_width=True):
                res = run_agent_query("discovery", disc_goal, st.session_state.active_groq_model)
                if res: st.success(res)

    # LEFT COLUMN: INTERACTIVE RESEARCH
    with left_col:
        with st.container(border=True):
            st.markdown("##### Scientific Intelligence Log")
            chat_box = st.container(height=350)
            
            for msg in st.session_state.ai_chat_history:
                with chat_box.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if chat_input := st.chat_input("Query the Lead Scientist..."):
                st.session_state.ai_chat_history.append({"role": "user", "content": chat_input})
                with chat_box.chat_message("user"): st.markdown(chat_input)
                
                with chat_box.chat_message("assistant"):
                    with st.spinner("Analyzing Chemical Space (at 800 tokens/sec)..."):
                        response = run_agent_query("chat", chat_input, st.session_state.active_groq_model)
                        if response:
                            st.markdown(response)
                            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})

        with st.expander("AI Experiment Planner", expanded=False):
            plan_goal = st.text_input("Research Objective:", placeholder="e.g. Find a non-toxic replacement for Lead-halide perovskites")
            if st.button("Generate Workflow", type="primary", use_container_width=True):
                res = run_agent_query("planner", plan_goal, st.session_state.active_groq_model)
                if res: st.info(res)

        with st.expander("  Failure Diagnostic Engine", expanded=False):
            fail_desc = st.text_area("Observations:", placeholder="Cracking observed during sintering at 1200K.")
            if st.button("Diagnose Failure", use_container_width=True):
                res = run_agent_query("diagnostic", fail_desc, st.session_state.active_groq_model)
                if res: st.error(res)


    if st.session_state.active_mat:
        with st.status(" Monitoring Cross-Tab Intelligence..."):
            is_metal = st.session_state.active_mat.get('is_metal', False)
            if str(is_metal).lower() not in ['true', 'metal', '1']:
                st.write("  **Agent Note:** Material is Non-Metallic. UV-Vis in Tab 4 should reflect transparency.")
            else:
                st.write("  **Agent Note:** Material is Metallic. Plasma frequency overrides visible optics.")

# TAB 4 

with tab4:
    st.markdown("### Thermodynamic Stability & Optics")
    st.caption(" 319k Thermodynamic and 7.3k Dielectric databases are used ")

    if st.session_state.active_mat:
        mat = st.session_state.active_mat
        mat_id = mat.get('material_id', '')
        c1, c2 = st.columns(2)


        df_thermo = pd.DataFrame()
        if os.path.exists('db_thermo.sqlite'):
            try:
                with contextlib.closing(sqlite3.connect('db_thermo.sqlite')) as conn:
                    cols_info = conn.execute("PRAGMA table_info(thermodynamics)").fetchall()
                    existing_cols = [c[1] for c in cols_info]
                    id_col = "material_id" if "material_id" in existing_cols else "mp_id" if "mp_id" in existing_cols else None
                    
                    if id_col:
                        df_thermo = pd.read_sql_query(f"SELECT * FROM thermodynamics WHERE {id_col}=?", conn, params=(mat_id,))
            except Exception as e:
                st.error(f"Thermo SQL Error: {e}")

        with c1:
            st.markdown("#### Structural Density")
            density = "N/A"

            if not df_thermo.empty and "density" in df_thermo.columns:
                val = np.ravel(df_thermo["density"])[0]
                if pd.notna(val): density = float(val)

            if density == "N/A":
                val = mat.get('density')
                if val is not None: density = float(val)

            st.divider()
            if density != "N/A":
                st.metric("Theoretical Density", f"{density:.3f} g/cm³")
            else:
                st.metric("Theoretical Density", "N/A")


        optics_res = None
        if os.path.exists('db_advanced_physics.sqlite'):
            try:
                with contextlib.closing(sqlite3.connect('db_advanced_physics.sqlite')) as conn:
                    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                    if tables:
                        t_name = "dielectrics" if "dielectrics" in tables else tables[0]
                        cols_info = conn.execute(f"PRAGMA table_info({t_name})").fetchall()
                        existing_cols = [c[1] for c in cols_info]
                        id_col = "material_id" if "material_id" in existing_cols else "mp_id" if "mp_id" in existing_cols else "task_id" if "task_id" in existing_cols else None
                        
                        if id_col:
                            optics_res = conn.execute(f"SELECT refractive_index, dielectric_const FROM {t_name} WHERE {id_col}=?", (mat_id,)).fetchone()
            except Exception as e:
                pass 

        with c2:
            st.markdown("#### Optics Engine")
            

            db_n = float(optics_res[0]) if optics_res and optics_res[0] is not None else None
            db_eps = float(optics_res[1]) if optics_res and optics_res[1] is not None else None
            
            if db_n and db_eps:
                st.metric("Refractive Index (n)", f"{db_n:.3f}", "Verified DB Match")
                st.metric("Dielectric Constant (ε)", f"{db_eps:.2f}", "Verified DB Match")
            else:

                raw_bg = mat.get('band_gap')
                
                if pd.notna(raw_bg) and float(raw_bg) > 0.1:
                    bg_ev = float(raw_bg)
                    est_n = (95.0 / bg_ev)**0.25
                    est_eps = est_n**2 
                    
                    st.metric("Refractive Index (n)", f"{est_n:.3f}", "Moss Relation Estimate", delta_color="off")
                    st.metric("Dielectric Constant (ε_∞)", f"{est_eps:.2f}", f"From Bandgap ({bg_ev:.2f} eV)", delta_color="off")
                else:
                    is_metal = str(mat.get('is_metal', 'False')).lower() in ['true', '1']

                    if is_metal or (pd.notna(raw_bg) and float(raw_bg) <= 0.1):
                        st.metric("Refractive Index (n)", "Opaque", "Metallic / Zero Bandgap", delta_color="off")
                        st.metric("Dielectric Constant", "Infinite", "Metallic Shielding", delta_color="off")
                    else:
                        st.metric("Refractive Index (n)", "N/A", "Bandgap data missing", delta_color="off")
                        st.metric("Dielectric Constant", "N/A", "Bandgap data missing", delta_color="off")
                
                st.caption(f"ID {mat_id} missing from Optics DB. Approximated via empirical physics.")
    else:
        st.info(" Use the Global Search in the sidebar to lock onto a material first.")

    st.divider()
    st.markdown("### Dynamic Thermal Simulation")
    st.caption(" **Simulation Note:** This plot provides a textbook visual representation of the Dulong-Petit high-temperature limit (Cp ≈ 3nR). The low-temperature Debye curve is mathematically approximated for demonstration and does not use live phonon database scattering.")
    max_temp = st.slider("Select Maximum Temperature (Kelvin)", 100, 3000, 1500, 100)
    
    target_list = []
    if st.session_state.active_mat:
        target_list.append(st.session_state.active_mat['formula'])
    
    if not st.session_state.batch_data.empty:

        batch_f = [f for f in st.session_state.batch_data["Formula"].tolist() if f not in target_list]
        target_list.extend(batch_f)


    if not target_list:
        target_list = ["LiCoO2", "Si"]

    temperatures = np.linspace(10, max_temp, 50)
    thermal_data = []

    for form in target_list:

        try:
            n_atoms = Composition(form).num_atoms
        except:
            n_atoms = 1


        limit_cp = 24.94 * n_atoms
        

        pseudo_debye = 200 + (sum(ord(c) for c in form) * 7 % 300)
        

        hc = limit_cp * (1 - np.exp(-temperatures / pseudo_debye))
        hc += np.random.normal(0, limit_cp * 0.01, len(temperatures))
        hc = np.maximum(0, hc) 

        for t, h in zip(temperatures, hc):
            thermal_data.append({"Formula": form, "Temperature (K)": t, "Heat Capacity (J/mol·K)": h})

    fig_thermo = px.line(pd.DataFrame(thermal_data), x="Temperature (K)", y="Heat Capacity (J/mol·K)", color="Formula")
    fig_thermo.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_thermo.update_traces(line=dict(width=3)) 
    st.plotly_chart(fig_thermo, use_container_width=True)


    st.divider()
    st.markdown("#### Fundamental Physics Engine")
    st.caption("Strictly analytical models driven by universal constants. No empirical presets or approximations.")

    physics_module = st.selectbox("Select Analytical Model:", [
        "1. Thermal Emission (Planck's Blackbody Law)",
        "2. Optical Reflectance (Fresnel Equations)"
    ])

    if "Planck" in physics_module:
        st.markdown("**Exact Spectral Radiance & Emissive Power**")
        st.latex(r"B(\lambda, T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{\frac{hc}{\lambda k_B T}} - 1}")

        temp_k = st.number_input("Absolute Temperature $T$ (Kelvin):", min_value=1.0, value=3000.0, step=100.0, key="planck_t")
        
        if st.button("Calculate Thermal Emission Spectrum", use_container_width=True):
            h = 6.626e-34
            c = 3.0e8
            k_B = 1.38e-23
            

            waves_nm = np.linspace(100, 5000, 500)
            waves_m = waves_nm * 1e-9
            

            power_term = (h * c) / (waves_m * k_B * temp_k)
            power_term = np.clip(power_term, a_min=None, a_max=700) 
            

            radiance = (2 * h * c**2) / (waves_m**5 * (np.exp(power_term) - 1))
            

            radiance_kw_nm = radiance * 1e-9 / 1000.0
            
            wien_peak_m = 2.8977719e-3 / temp_k
            total_power = 5.67e-8 * temp_k**4
            
            fig_planck = px.line(x=waves_nm, y=radiance_kw_nm, title=f"Blackbody Radiation Spectrum at {temp_k} K", labels={'x': 'Wavelength (nm)', 'y': 'Spectral Radiance (kW / m²·sr·nm)'})
            fig_planck.add_vline(x=wien_peak_m * 1e9, line_dash="dash", line_color="#F56565", annotation_text=f"Peak: {wien_peak_m * 1e9:.1f} nm")
            fig_planck.update_traces(line_color='#00E676', fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)')
            st.plotly_chart(fig_planck, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Peak Emission Wavelength ($\lambda_{max}$)", f"{wien_peak_m * 1e9:.1f} nm", "Wien's Displacement Law")
            c2.metric("Total Emissive Power ($j^*$)", f"{total_power / 1000.0:,.1f} kW/m²", "Stefan-Boltzmann Law")

    elif "Fresnel" in physics_module:
        st.markdown("**Exact Interfacial Reflectance & Transmittance**")
        st.latex(r"R_s = \left| \frac{n_1 \cos\theta_i - n_2 \cos\theta_t}{n_1 \cos\theta_i + n_2 \cos\theta_t} \right|^2")
        
        c1, c2 = st.columns(2)
        n1 = c1.number_input("Refractive Index of Medium 1 ($n_1$):", min_value=1.0, value=1.0, format="%.3f", help="e.g., Air = 1.0", key="fresnel_n1")
        

        n2 = c2.number_input("Refractive Index of Medium 2 ($n_2$):", min_value=0.01, value=1.5, format="%.3f", help="e.g., Glass = 1.5", key="fresnel_n2")
        
        if st.button("Calculate Fresnel Reflection Curves", use_container_width=True):
            angles_deg = np.linspace(0, 90, 400)
            angles_rad = np.radians(angles_deg)
            
            R_s, R_p = np.zeros_like(angles_rad), np.zeros_like(angles_rad)
            
            for i, theta_i in enumerate(angles_rad):

                sin_theta_t = (n1 / n2) * np.sin(theta_i)
                
                if sin_theta_t >= 1.0:

                    R_s[i], R_p[i] = 1.0, 1.0
                else:
                    theta_t = np.arcsin(sin_theta_t)
                    

                    rs = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
                    rp = (n1 * np.cos(theta_t) - n2 * np.cos(theta_i)) / (n1 * np.cos(theta_t) + n2 * np.cos(theta_i))
                    R_s[i], R_p[i] = rs**2, rp**2
            
            df_fresnel = pd.DataFrame({
                "Incident Angle (°)": angles_deg,
                "S-Polarized ($R_s$)": R_s,
                "P-Polarized ($R_p$)": R_p,
                "Unpolarized Average": (R_s + R_p) / 2.0
            })
            
            fig_fres = px.line(df_fresnel, x="Incident Angle (°)", y=["S-Polarized ($R_s$)", "P-Polarized ($R_p$)", "Unpolarized Average"], title=f"Fresnel Reflectance ($n_1={n1} \\rightarrow n_2={n2}$)")
            fig_fres.update_layout(yaxis_title="Reflectance (0 to 1)", yaxis_range=[0, 1.05], legend_title="Polarization")
            

            if n2 > n1:
                brewster_deg = np.degrees(np.arctan(n2 / n1))
                fig_fres.add_vline(x=brewster_deg, line_dash="dash", line_color="white", annotation_text=f"Brewster's Angle: {brewster_deg:.1f}°")
            elif n1 > n2:
                critical_deg = np.degrees(np.arcsin(n2 / n1))
                fig_fres.add_vline(x=critical_deg, line_dash="dash", line_color="white", annotation_text=f"Critical Angle (TIR): {critical_deg:.1f}°")
                
            st.plotly_chart(fig_fres, use_container_width=True)


    st.divider()
    st.markdown("#### Optics & Photonics Simulator")
    bg_ev = st.number_input("Theoretical Band-Gap Energy (eV):", min_value=0.1, max_value=10.0, value=2.1, step=0.1)
    if st.button("Simulate Optical Properties"):
        wavelength = 1240.0 / bg_ev


        def wave_to_hex(w):
            if w < 380:
                return "#333333", "Ultraviolet (Invisible)"
            elif w < 450:
                return "#8A2BE2", "Violet"
            elif w < 495:
                return "#0000FF", "Blue"
            elif w < 570:
                return "#00FF00", "Green"
            elif w < 590:
                return "#FFFF00", "Yellow"
            elif w < 620:
                return "#FFA500", "Orange"
            elif w < 750:
                return "#FF0000", "Red"
            else:
                return "#333333", "Infrared (Invisible)"


        color_hex, color_name = wave_to_hex(wavelength)
        c1_opt, c2_opt = st.columns(2)
        c1_opt.metric("Absorption Edge (Wavelength)", f"{wavelength:.1f} nm")
        c2_opt.markdown(f"**Visual Color:** {color_name}")
        c2_opt.markdown(
            f'<div style="background-color: {color_hex}; width: 100%; height: 50px; border-radius: 8px; border: 2px solid #555;"></div>',
            unsafe_allow_html=True)

        x_waves = np.linspace(200, 1000, 400)
        y_abs = np.where(x_waves <= wavelength, 1 - np.exp(-(wavelength - x_waves) / 50), 0.05) + np.random.normal(0,
                                                                                                                   0.015,
                                                                                                                   400)
        fig_uv = px.line(x=x_waves, y=y_abs, labels={'x': 'Wavelength (nm)', 'y': 'Absorbance (a.u.)'},
                         title="Simulated UV-Vis Spectrum")
        fig_uv.add_vline(x=wavelength, line_dash="dash", line_color="red", annotation_text="Band-Gap Edge")
        fig_uv.update_traces(line_color='#00E676')
        st.plotly_chart(fig_uv, use_container_width=True)

# TAB 5 

with tab5:
    st.markdown("### Equations Library")

    formula_choice = st.selectbox("Select Engineering Equation:", [
        "1. Bragg's Law (X-Ray Diffraction)", "2. Hall-Petch (Yield Strength)", "3. Arrhenius Equation (Reaction Rate)",
        "4. Scherrer Equation (Crystallite Size)", "5. Gibbs Free Energy (Thermodynamics)",
        "6. Hooke's Law (Stress & Strain)",
        "7. Linear Thermal Expansion", "8. Fick's First Law (Diffusion Flux)", "9. Nernst Equation (Electrochemistry)",
        "10. Theoretical Density (Unit Cell)", "11. Planck-Einstein (Photon Energy)", "12. Electrical Resistivity",
        "13. Cubic Lattice Parameter (a from d)", "14. Snell's Law (Refractive Index)", "15. Charge Carrier Mobility",
        "16. Goldschmidt Tolerance (Perovskites)", "17. Battery Specific Capacity (mAh/g)",
        "18. Bravais Lattice & Crystal System"
    ])



    if "Bragg" in formula_choice:
        st.latex(r"n\lambda = 2d \sin(\theta)")
        c1, c2, c3 = st.columns(3)
        n_order = c1.number_input("Order (n)", value=1.0, format="%g")
        wavelength = c2.number_input("Wavelength λ (nm)", value=0.154, format="%g", key="wave_bragg")
        theta = c3.number_input("Angle θ (degrees)", value=20.0, format="%g")

        if st.button("Calculate d-spacing", key="calc_d_tab5"):
            if theta <= 0 or theta >= 180:
                st.error("Angle θ must be between 0° and 180° (exclusive).")
            else:
                d = (n_order * wavelength) / (2 * math.sin(math.radians(theta)))
                st.success(f"**Interplanar Spacing (d):** {d:.4f} nm")



    elif "Hall-Petch" in formula_choice:
        st.latex(r"\sigma_y = \sigma_0 + \frac{k_y}{\sqrt{d}}")
        c1, c2, c3 = st.columns(3)
        sigma_0 = c1.number_input("Friction Stress σ₀ (MPa)", value=50.0, format="%g")
        k_y = c2.number_input("Locking Parameter k (MPa·m^0.5)", value=0.5, format="%g")
        grain_size = c3.number_input("Grain Size d (m)", value=0.0001, min_value=1e-10, format="%g")
        if st.button("Calculate Yield Strength", key="calc_ys_tab5"):
            st.success(f"**Yield Strength (σ_y):** {sigma_0 + (k_y / math.sqrt(grain_size)):.2f} MPa")

    elif "Arrhenius" in formula_choice:
        st.latex(r"k = A \exp\left(-\frac{E_a}{RT}\right)")
        c1, c2, c3 = st.columns(3)
        pre_exp = c1.number_input("Pre-exponential A", value=1e13, format="%g")
        e_a = c2.number_input("Activation Energy Ea (J/mol)", value=50000.0, format="%g")
        temp = c3.number_input("Temperature T (K)", value=300.0, min_value=0.01, format="%g")
        if st.button("Calculate Rate Constant (k)"): st.success(
            f"**Rate Constant (k):** {pre_exp * math.exp(-e_a / (8.314 * temp)):.4e}")

    elif "Scherrer" in formula_choice:
        st.latex(r"\tau = \frac{K\lambda}{\beta \cos(\theta)}")
        c1, c2, c3 = st.columns(3)
        shape_factor = c1.number_input("Shape Factor (K)", value=0.9, format="%g")
        wavelength = c2.number_input("Wavelength λ (nm)", value=0.154, format="%g", key="wave_scherrer")
        beta = c3.number_input("Line Broadening β (rad)", value=0.005, min_value=1e-5, format="%g")
        theta = st.number_input("Bragg Angle θ (degrees)", value=20.0, max_value=89.9, format="%g")
        if st.button("Calculate Crystallite Size"): st.success(
            f"**Crystallite Size (τ):** {(shape_factor * wavelength) / (beta * math.cos(math.radians(theta))):.4f} nm")

    elif "Gibbs" in formula_choice:
        st.latex(r"\Delta G = \Delta H - T\Delta S")
        c1, c2, c3 = st.columns(3)
        delta_h = c1.number_input("Enthalpy ΔH (J)", value=-15000.0, format="%g")
        temp = c2.number_input("Temperature T (K)", value=298.15, format="%g")
        delta_s = c3.number_input("Entropy ΔS (J/K)", value=50.0, format="%g")
        if st.button("Calculate Gibbs Free Energy"): st.success(f"**ΔG:** {delta_h - (temp * delta_s):.2f} J")

    elif "Hooke" in formula_choice:
        st.latex(r"\sigma = E \cdot \epsilon")
        c1, c2 = st.columns(2)
        modulus = c1.number_input("Young's Modulus E (GPa)", value=200.0, format="%g")
        strain = c2.number_input("Strain ε (unitless)", value=0.002, format="%g")
        if st.button("Calculate Stress"): st.success(f"**Stress (σ):** {modulus * 1000 * strain:.2f} MPa")

    elif "Thermal Expansion" in formula_choice:
        st.latex(r"\Delta L = \alpha L_0 \Delta T")
        c1, c2, c3 = st.columns(3)
        alpha = c1.number_input("Expansion Coeff α (1/K)", value=1.2e-5, format="%g")
        length = c2.number_input("Original Length L₀ (m)", value=1.0, format="%g")
        delta_t = c3.number_input("Temp Change ΔT (K)", value=50.0, format="%g")
        if st.button("Calculate Length Change"): st.success(
            f"**Change in Length (ΔL):** {(alpha * length * delta_t) * 1000:.4f} mm")

    elif "Fick" in formula_choice:
        st.latex(r"J = -D \frac{dc}{dx}")
        c1, c2, c3 = st.columns(3)
        diff_coeff = c1.number_input("Diffusion Coeff D (m²/s)", value=1e-10, format="%g")
        delta_c = c2.number_input("Concentration Diff dc (mol/m³)", value=50.0, format="%g")
        delta_x = c3.number_input("Distance dx (m)", value=0.001, min_value=1e-9, format="%g")
        if st.button("Calculate Flux"): st.success(
            f"**Diffusion Flux (J):** {abs(-diff_coeff * (delta_c / delta_x)):.4e} mol/(m²·s)")

    elif "Nernst" in formula_choice:
        st.latex(r"E = E^0 - \frac{0.0592}{n} \log_{10}(Q)")
        c1, c2, c3 = st.columns(3)
        e_zero = c1.number_input("Standard Potential E⁰ (V)", value=1.10, format="%g")
        electrons = c2.number_input("Moles of Electrons (n)", value=2.0, min_value=0.001, format="%g")
        q_quotient = c3.number_input("Reaction Quotient (Q)", value=0.01, min_value=1e-9, format="%g")
        if st.button("Calculate Potential"): st.success(
            f"**Cell Potential (E):** {e_zero - ((0.0592 / electrons) * math.log10(q_quotient)):.4f} V")

    elif "Theoretical Density" in formula_choice:
        st.latex(r"\rho = \frac{n A}{V_c N_A}")
        c1, c2, c3 = st.columns(3)
        n_atoms = c1.number_input("Atoms per unit cell (n)", value=4.0, format="%g")
        atomic_weight = c2.number_input("Atomic Weight A (g/mol)", value=63.55, format="%g")
        vol_angstrom = c3.number_input("Unit Cell Vol (Å³)", value=47.3, min_value=1e-9, format="%g")
        if st.button("Calculate Density"): st.success(
            f"**Theoretical Density (ρ):** {(n_atoms * atomic_weight) / ((vol_angstrom * 1e-24) * 6.022e23):.4f} g/cm³")

    elif "Planck" in formula_choice:
        st.latex(r"E = \frac{hc}{\lambda}")
        wavelength = st.number_input("Wavelength λ (nm)", value=500.0, min_value=1e-9, format="%g", key="wave_planck")
        if st.button("Calculate Energy"): st.success(
            f"**Energy:** {((6.626e-34 * 3.0e8) / (wavelength * 1e-9)) / 1.602e-19:.2f} eV  |  {((6.626e-34 * 3.0e8) / (wavelength * 1e-9)):.4e} J")

    elif "Resistivity" in formula_choice:
        st.latex(r"\rho = R \frac{A}{L}")
        c1, c2, c3 = st.columns(3)
        resistance = c1.number_input("Resistance R (Ω)", value=0.5, format="%g")
        area = c2.number_input("Cross-sectional Area A (m²)", value=1e-6, format="%g")
        length = c3.number_input("Length L (m)", value=2.0, min_value=1e-9, format="%g")
        if st.button("Calculate Resistivity"): st.success(
            f"**Resistivity (ρ):** {resistance * (area / length):.4e} Ω·m")

    elif "Cubic Lattice" in formula_choice:
        st.latex(r"a = d \sqrt{h^2 + k^2 + l^2}")
        c1, c2 = st.columns(2)
        d_space = c1.number_input("d-spacing (nm)", value=0.203, format="%g")
        miller = c2.text_input("Miller Indices (h k l)", value="1 1 1")
        if st.button("Calculate a", key="btn_cubic_a"):
            try:
                parts = miller.replace(',', ' ').split()
                if len(parts) != 3:
                    raise ValueError
                h, k, l = map(int, parts)
                st.success(f"**Lattice Parameter (a):** {d_space * math.sqrt(h**2 + k**2 + l**2):.4f} nm")
            except ValueError:
                st.error("Enter exactly 3 integers separated by spaces — e.g. '1 1 1'.")

    elif "Snell" in formula_choice:
        st.latex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)")
        c1, c2, c3 = st.columns(3)
        n1 = c1.number_input("Refractive Index 1 (n₁)", value=1.0, format="%g")
        theta1 = c2.number_input("Incident Angle θ₁ (deg)", value=45.0, format="%g")
        n2 = c3.number_input("Refractive Index 2 (n₂)", value=1.5, min_value=0.001, format="%g")
        if st.button("Calculate Refracted Angle"):
            sin_theta2 = (n1 * math.sin(math.radians(theta1))) / n2
            st.success(
                f"**Refracted Angle (θ₂):** {math.degrees(math.asin(sin_theta2)):.2f}°" if sin_theta2 <= 1 else "Total Internal Reflection.")

    elif "Mobility" in formula_choice:
        st.latex(r"\mu = \frac{v_d}{E}")
        c1, c2 = st.columns(2)
        v_drift = c1.number_input("Drift Velocity v_d (m/s)", value=1000.0, format="%g")
        e_field = c2.number_input("Electric Field E (V/m)", value=10000.0, min_value=1e-9, format="%g")
        if st.button("Calculate Mobility"): st.success(f"**Mobility (μ):** {v_drift / e_field:.4f} m²/(V·s)")

    elif "Goldschmidt" in formula_choice:
        st.latex(r"t = \frac{r_A + r_X}{\sqrt{2}(r_B + r_X)}")
        st.markdown("**Predict Perovskite ($ABX_3$) Lattice Stability**")
        c1, c2, c3 = st.columns(3)
        r_A = c1.number_input("A-site Radius ($r_A$) in Å", value=1.44, format="%g")
        r_B = c2.number_input("B-site Radius ($r_B$) in Å", value=0.605, format="%g")
        r_X = c3.number_input("Anion Radius ($r_X$) in Å", value=1.40, format="%g")
        if st.button("Calculate Tolerance Factor (t)"):
            if (r_B + r_X) == 0:
                st.error("r_B + r_X cannot be zero — check your ionic radii.")
            else:
                t_factor = (r_A + r_X) / (math.sqrt(2) * (r_B + r_X))
                st.success(f"**Tolerance Factor (t):** {t_factor:.4f}")
                if 0.9 <= t_factor <= 1.0:
                    st.info("Structure: **Ideal Cubic**")
                elif 0.71 <= t_factor < 0.9:
                    st.warning("Structure: **Orthorhombic / Rhombohedral**")
                elif t_factor > 1.0:
                    st.warning("Structure: **Hexagonal**")
                else:
                    st.error("Structure: **Unstable**")

    elif "Battery" in formula_choice:
        st.latex(r"Q = \frac{nF}{3.6 M_w}")
        st.markdown("**Calculate Theoretical Specific Capacity**")
        c1, c2 = st.columns(2)
        batt_form = c1.text_input("Active Material Formula:", value="LiFePO4")
        n_electrons = c2.number_input("Electrons Transferred (n):", value=1.0)
        if st.button("Calculate Specific Capacity"):
            try:
                clean_batt = {"Graphite": "C6", "Silicon": "Si", "Lithium": "Li"}.get(batt_form, batt_form)
                mw = Composition(clean_batt).weight
                capacity = (n_electrons * 96485) / (3.6 * mw)
                st.info(f"  **Theoretical Specific Capacity:** {capacity:.1f} mAh/g")
            except:
                st.error("Invalid chemical formula.")

    elif "Bravais" in formula_choice:
        st.markdown("**Crystallographic System Classifier**")
        c1, c2, c3 = st.columns(3)
        a = c1.number_input("Length a (Å)", value=5.0)
        b = c2.number_input("Length b (Å)", value=5.0)
        c = c3.number_input("Length c (Å)", value=5.0)
        c4, c5, c6 = st.columns(3)
        alpha = c4.number_input("Angle α (°)", value=90.0)
        beta = c5.number_input("Angle β (°)", value=90.0)
        gamma = c6.number_input("Angle γ (°)", value=90.0)
        if st.button("Classify System"):
            tol = 0.01


            def eq(x, y):
                return abs(x - y) < tol


            system = "Unknown"
            if eq(a, b) and eq(b, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
                system = " Cubic"
            elif eq(a, b) and not eq(b, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
                system = " Tetragonal"
            elif not eq(a, b) and not eq(b, c) and not eq(a, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
                system = " Orthorhombic"
            elif eq(a, b) and not eq(b, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 120):
                system = " Hexagonal"
            elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma) and not eq(alpha, 90):
                system = " Rhombohedral"
            elif not eq(a, b) and not eq(b, c) and not eq(a, c) and eq(alpha, 90) and eq(gamma, 90) and not eq(beta,
                                                                                                               90):
                system = " Monoclinic"
            elif not eq(a, b) and not eq(b, c) and not eq(a, c) and not eq(alpha, beta) and not eq(beta,
                                                                                                   gamma) and not eq(
                alpha, 90):
                system = " Triclinic"
            st.success(f"**System Detected:** {system}")

    st.divider()
    st.markdown("###   Advanced Computational Engine")
    st.caption("Categorized advanced solvers for solid-state physics, quantum mechanics, chemistry, and numerical methods.")


    c_domain, c_calc = st.columns([1, 1.5])
    domain = c_domain.selectbox("Select Scientific Domain:", [
        "1. Crystallography & Structure", 
        "2. Quantum & Solid State Physics", 
        "3. Thermodynamics & Chemistry",
        "4. Transport & Fluid Dynamics",
        "5. Numerical & Mathematical Tools"
    ])


    if "Crystallography" in domain:
        adv_choice = c_calc.selectbox("Select Calculator:", ["Atomic Packing Factor (APF)", "Multi-Peak Bragg's Law", "Vegard's Law (Alloy Predictor)"])
    elif "Quantum" in domain:
        adv_choice = c_calc.selectbox("Select Calculator:", ["Particle-in-a-Box Solver", "Bohr Model Energy levels", "De Broglie Wavelength", "Heisenberg Uncertainty", "Fermi Energy (Free Electron)"])
    elif "Thermodynamics" in domain:
        adv_choice = c_calc.selectbox("Select Calculator:", ["Clausius-Clapeyron (Phase Temp/Pressure)", "Entropy of Mixing (Ideal Solution)", "pH & Buffer (Henderson-Hasselbalch)", "Limiting Reagent & Yield"])
    elif "Transport" in domain:
        adv_choice = c_calc.selectbox("Select Calculator:", ["Dimensionless Numbers (Reynolds & Prandtl)"])
    elif "Numerical" in domain:
        adv_choice = c_calc.selectbox("Select Calculator:", ["Numerical Integration (Trapezoidal)", "Root Finder (Bisection Method)", "Error Propagation"])





    if adv_choice == "Atomic Packing Factor (APF)":
        st.markdown("**Calculate the packing efficiency of standard unit cells.**")
        lattice_type = st.radio("Lattice Type:", ["Simple Cubic (SC)", "Body-Centered Cubic (BCC)", "Face-Centered Cubic (FCC)", "Hexagonal Close-Packed (HCP)"], horizontal=True)
        if st.button("Calculate APF"):
            if "SC" in lattice_type: apf, atoms, relation = 0.52, 1, "a = 2r"
            elif "BCC" in lattice_type: apf, atoms, relation = 0.68, 2, "a = 4r / √3"
            elif "FCC" in lattice_type: apf, atoms, relation = 0.74, 4, "a = 2r√2"
            elif "HCP" in lattice_type: apf, atoms, relation = 0.74, 6, "a = 2r, c ≈ 1.633a"
            st.success(f"**Packing Efficiency (APF):** {apf * 100}%")
            st.info(f"**Atoms per Unit Cell:** {atoms} | **Lattice-Radius Relation:** {relation}")

    elif adv_choice == "Multi-Peak Bragg's Law":
        st.latex(r"\lambda = 2d \sin(\theta)")
        c1, c2 = st.columns(2)
        d_space_adv = c1.number_input("Interplanar Spacing d (nm):", value=0.203, min_value=1e-9, format="%g")
        wave_adv = c2.number_input("X-Ray Wavelength λ (nm):", value=0.15406, format="%g") 
        if st.button("Calculate Visible Peaks"):
            peaks = []
            for n in range(1, 6):
                sin_theta = (n * wave_adv) / (2 * d_space_adv)
                if sin_theta <= 1.0:
                    theta = math.degrees(math.asin(sin_theta))
                    peaks.append({"Order (n)": n, "Theta (θ)": round(theta, 2), "2-Theta (2θ)": round(theta * 2, 2)})
            if peaks:
                st.table(pd.DataFrame(peaks))
            else:
                st.error("No valid diffraction peaks (sin(θ) > 1). Wavelength is too large for this d-spacing.")

    elif adv_choice == "Vegard's Law (Alloy Predictor)":
        st.latex(r"a_{alloy} = x a_A + (1-x) a_B")
        st.markdown("**Predict the lattice parameter of a solid solution.**")
        c1, c2, c3 = st.columns(3)
        a1 = c1.number_input("Lattice Param A (Å):", value=3.61, help="e.g., Copper")
        a2 = c2.number_input("Lattice Param B (Å):", value=4.08, help="e.g., Gold")
        x_frac = c3.slider("Mole Fraction of A (x):", 0.0, 1.0, 0.5)
        if st.button("Predict Alloy Lattice"):
            a_alloy = (x_frac * a1) + ((1 - x_frac) * a2)
            st.success(f"**Predicted Lattice Parameter:** {a_alloy:.4f} Å")


    elif adv_choice == "Particle-in-a-Box Solver":
        st.latex(r"E_n = \frac{n^2 h^2}{8 m L^2}")
        c1, c2, c3 = st.columns(3)
        n_level = c1.number_input("Quantum State (n):", min_value=1, value=1, step=1)
        box_L_nm = c2.number_input("Box Length L (nm):", value=1.0, format="%g")
        p_mass = c3.selectbox("Particle:", ["Electron", "Proton", "Neutron"])
        if st.button("Solve Energy State"):
            h = 6.626e-34
            m = {"Electron": 9.109e-31, "Proton": 1.672e-27, "Neutron": 1.674e-27}[p_mass]
            L = box_L_nm * 1e-9 
            e_joules = ( (n_level**2) * (h**2) ) / ( 8 * m * (L**2) )
            st.success(f"**Energy level $E_{n_level}$:** {e_joules / 1.602e-19:.4f} eV")

    elif adv_choice == "Bohr Model Energy levels":
        st.latex(r"E_n = -13.6 \frac{Z^2}{n^2} \text{ eV}, \quad r_n = 0.529 \frac{n^2}{Z} \text{ \AA}")
        c1, c2 = st.columns(2)
        z_val = c1.number_input("Atomic Number (Z):", min_value=1, value=1, step=1)
        n_shell = c2.number_input("Principal Quantum Number (n):", min_value=1, value=1, step=1)
        if st.button("Calculate Bohr Orbit"):
            st.success(f"**Energy ($E_n$):** {-13.6 * ((z_val**2) / (n_shell**2)):.3f} eV")
            st.info(f"**Orbital Radius ($r_n$):** {0.529 * ((n_shell**2) / z_val):.4f} Å")

    elif adv_choice == "De Broglie Wavelength":
        st.latex(r"\lambda = \frac{h}{p} = \frac{h}{mv}")
        c1, c2 = st.columns(2)
        mass = c1.number_input("Particle Mass (kg):", value=9.109e-31, format="%e", help="Default is Electron")
        vel = c2.number_input("Velocity (m/s):", value=1.0e6, format="%e")
        if st.button("Calculate Wavelength"):
            if vel > 0:
                wave_m = 6.626e-34 / (mass * vel)
                st.success(f"**Wavelength ($\lambda$):** {wave_m * 1e9:.4e} nm  |  {wave_m * 1e10:.4f} Å")
            else: st.error("Velocity must be > 0.")

    elif adv_choice == "Heisenberg Uncertainty":
        st.latex(r"\Delta x \Delta p \ge \frac{\hbar}{2}")
        c1, c2 = st.columns(2)
        mode = c1.radio("Known Uncertainty:", ["Position (Δx)", "Momentum (Δp)"])
        val = c2.number_input("Value (in SI units):", value=1e-9, format="%e")
        if st.button("Calculate Minimum Uncertainty"):
            hbar = 1.054e-34
            min_uncert = (hbar / 2) / val
            target = "Momentum (Δp) in kg·m/s" if "Position" in mode else "Position (Δx) in meters"
            st.success(f"**Minimum uncertainty in {target}:** {min_uncert:.4e}")

    elif adv_choice == "Fermi Energy (Free Electron)":
        st.latex(r"E_F = \frac{\hbar^2}{2m_e} (3\pi^2 n)^{2/3}")
        n_density = st.number_input("Electron Density $n$ (electrons/cm³):", value=8.49e22, format="%e", help="Default is Cu")
        if st.button("Calculate Fermi Energy"):
            hbar, m_e = 1.054e-34, 9.109e-31
            n_m3 = n_density * 1e6
            e_f_j = ( (hbar**2) / (2 * m_e) ) * ( (3 * math.pi**2 * n_m3)**(2/3) )
            st.success(f"**Fermi Energy ($E_F$):** {e_f_j / 1.602e-19:.4f} eV")


    elif adv_choice == "Clausius-Clapeyron (Phase Temp/Pressure)":
        st.latex(r"\ln\left(\frac{P_2}{P_1}\right) = -\frac{\Delta H_{vap}}{R} \left(\frac{1}{T_2} - \frac{1}{T_1}\right)")
        c1, c2, c3, c4 = st.columns(4)
        t1 = c1.number_input("Temp 1 (K):", value=373.15)
        p1 = c2.number_input("Pressure 1 (atm):", value=1.0)
        dh = c3.number_input("ΔH_vap (J/mol):", value=40650.0)
        t2 = c4.number_input("Temp 2 (K):", value=350.0)
        if st.button("Calculate New Pressure ($P_2$)"):
            ln_p2_p1 = -(dh / 8.314) * ((1/t2) - (1/t1))
            st.success(f"**New Vapor Pressure ($P_2$):** {p1 * math.exp(ln_p2_p1):.4f} atm")

    elif adv_choice == "Entropy of Mixing (Ideal Solution)":
        st.latex(r"\Delta S_{mix} = -R \sum x_i \ln(x_i)")
        c1, c2 = st.columns(2)
        moles_A = c1.number_input("Moles of A:", min_value=0.0, value=1.0)
        moles_B = c2.number_input("Moles of B:", min_value=0.0, value=1.0)
        if st.button("Calculate Mixing Entropy"):
            tot = moles_A + moles_B
            if tot > 0:
                ent = 0.0
                for m in [moles_A, moles_B]:
                    if m > 0: ent += (m/tot) * math.log(m/tot)
                st.success(f"**Molar Entropy of Mixing ($\Delta S_{{mix}}$):** {-8.314 * ent:.4f} J/(mol·K)")

    elif adv_choice == "pH & Buffer (Henderson-Hasselbalch)":
        st.latex(r"pH = pK_a + \log_{10}\left(\frac{[A^-]}{[HA]}\right)")
        c1, c2, c3 = st.columns(3)
        pka = c1.number_input("Acid pKa:", value=4.76, help="Default: Acetic Acid")
        base_c = c2.number_input("Base Concentration [A-]:", value=0.1, min_value=1e-9)
        acid_c = c3.number_input("Acid Concentration [HA]:", value=0.1, min_value=1e-9)
        if st.button("Calculate pH"):
            if acid_c > 0:
                st.success(f"**Buffer pH:** {pka + math.log10(base_c / acid_c):.3f}")
            else: st.error("Acid concentration must be > 0")

    elif adv_choice == "Limiting Reagent & Yield":
        st.markdown("**Analyze reaction stoichiometry: $aA + bB \\rightarrow cC$**")
        c1, c2, c3 = st.columns(3)
        with c1.container():
            st.markdown("**Reactant A**")
            coef_a = st.number_input("Coefficient (a)", min_value=1, value=1)
            moles_a = st.number_input("Available Moles A", min_value=0.0, value=2.0)
        with c2.container():
            st.markdown("**Reactant B**")
            coef_b = st.number_input("Coefficient (b)", min_value=1, value=2)
            moles_b = st.number_input("Available Moles B", min_value=0.0, value=3.0)
        with c3.container():
            st.markdown("**Product C**")
            coef_c = st.number_input("Coefficient (c)", min_value=1, value=1)
        
        if st.button("Calculate Yield"):
            ratio_a = moles_a / coef_a
            ratio_b = moles_b / coef_b
            limit_reagent = "A" if ratio_a < ratio_b else "B" if ratio_b < ratio_a else "None (Stoichiometric match)"
            max_yield = min(ratio_a, ratio_b) * coef_c
            st.success(f"**Limiting Reagent:** {limit_reagent}")
            st.info(f"**Theoretical Yield of C:** {max_yield:.3f} moles")


    elif adv_choice == "Dimensionless Numbers (Reynolds & Prandtl)":
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.latex(r"Re = \frac{\rho v L}{\mu}")
            rho = st.number_input("Density ρ (kg/m³):", value=1000.0)
            vel = st.number_input("Velocity v (m/s):", value=2.0)
            length = st.number_input("Length L (m):", value=0.05)
            mu = st.number_input("Viscosity μ (Pa·s):", value=0.001)
            if st.button("Calculate Re"):
                if mu > 0:
                    re_num = (rho * vel * length) / mu
                    ftype = "Laminar" if re_num < 2300 else "Transitional" if re_num < 4000 else "Turbulent"
                    st.success(f"**Re:** {re_num:,.1f} ({ftype})")
        with c2.container(border=True):
            st.latex(r"Pr = \frac{c_p \mu}{k}")
            cp = st.number_input("Specific Heat Cp (J/kg·K):", value=4184.0)
            mu_pr = st.number_input("Viscosity μ (Pa·s):", value=0.001, key="mu2")
            k_cond = st.number_input("Thermal Cond. k (W/m·K):", value=0.6)
            if st.button("Calculate Pr"):
                if k_cond > 0: st.success(f"**Pr:** {(cp * mu_pr) / k_cond:.2f}")


    elif adv_choice == "Numerical Integration (Trapezoidal)":
        st.latex(r"\int_{a}^{b} f(x) dx \approx \frac{\Delta x}{2} \sum (f(x_i) + f(x_{i+1}))")
        st.markdown("**Integrate $f(x) = cx^n$ over limits $a$ to $b$.**")
        c1, c2, c3, c4 = st.columns(4)
        

        c_val = c1.number_input("Coefficient (c):", value=1.0, key="num_c")
        n_val = c2.number_input("Exponent (n):", value=2.0, key="num_n")
        a_lim = c3.number_input("Lower limit (a):", value=0.0, key="num_a")
        b_lim = c4.number_input("Upper limit (b):", value=10.0, key="num_b")
        pts = st.slider("Subdivisions (Grid points):", 10, 1000, 100, key="num_pts")
        
        if st.button("Compute Integral", key="btn_num_int"):
            x_arr = np.linspace(a_lim, b_lim, pts)
            

            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                y_arr = c_val * (np.power(x_arr, n_val))
                
            if np.isnan(y_arr).any():
                st.error("Math Error: The limits and exponent resulted in imaginary/undefined numbers (e.g., square root of a negative number). Adjust your bounds.")
            elif np.isinf(y_arr).any():
                st.error("Math Error: The integral approaches infinity (Asymptote encountered).")
            else:

                try:
                    area = np.trapezoid(y_arr, x=x_arr)
                except AttributeError:
                    area = np.trapz(y_arr, x=x_arr)
                    

                try:
                    if n_val == -1.0:
                        if a_lim * b_lim <= 0:
                            exact = "Undefined (Crosses asymptote at x=0)"
                        else:
                            exact = c_val * (math.log(abs(b_lim)) - math.log(abs(a_lim)))
                    else:

                        b_term = complex(b_lim)**(n_val+1)
                        a_term = complex(a_lim)**(n_val+1)
                        exact_cplx = (c_val / (n_val + 1)) * (b_term - a_term)

                        if abs(exact_cplx.imag) < 1e-9:
                            exact = exact_cplx.real
                        else:
                            exact = "Imaginary"
                except Exception:
                    exact = "N/A"
                    
                st.success(f"**Numerical Area:** {area:.4f}")
                if isinstance(exact, float):
                    st.caption(f"Exact Analytical Area: {exact:.4f}")
                else:
                    st.caption(f"Exact Analytical Area: {exact}")

    elif adv_choice == "Root Finder (Bisection Method)":
        st.markdown("**Find root of $f(x) = ax^2 + bx + c = 0$ between bounds.**")
        c1, c2, c3 = st.columns(3)
        
        a = c1.number_input("a:", value=1.0, key="rf_a")
        b = c2.number_input("b:", value=-5.0, key="rf_b")
        c = c3.number_input("c:", value=6.0, key="rf_c")
        low = st.number_input("Lower bound:", value=0.0, key="rf_low")
        high = st.number_input("Upper bound:", value=2.5, key="rf_high")
        
        if st.button("Find Root", key="btn_rf"):
            def f(x): return a*x**2 + b*x + c
            

            real_low = min(low, high)
            real_high = max(low, high)
            
            if f(real_low) * f(real_high) > 0:
                st.error("Function must have opposite signs at boundaries (Intermediate Value Theorem). Try expanding your bounds.")
            elif f(real_low) == 0:
                st.success(f"**Root found at x:** {real_low:.5f}")
            elif f(real_high) == 0:
                st.success(f"**Root found at x:** {real_high:.5f}")
            else:
                tol, max_iter = 1e-6, 100
                root = (real_low + real_high) / 2.0  
                for _ in range(max_iter):
                    mid = (real_low + real_high) / 2.0
                    root = mid  
                    if abs(f(mid)) < tol: break
                    if f(real_low) * f(mid) < 0: real_high = mid
                    else: real_low = mid
                st.success(f"**Root found at x:** {root:.5f}")

    elif adv_choice == "Error Propagation":
        st.latex(r"\sigma_f = |f| \sqrt{\left(\frac{\sigma_x}{x}\right)^2 + \left(\frac{\sigma_y}{y}\right)^2}")
        st.markdown("**Calculate uncertainty for multiplication/division: $f = x \cdot y$ or $f = x / y$**")
        c1, c2 = st.columns(2)
        
        x_val = c1.number_input("Value x:", value=10.0, key="ep_x")
        x_err = c1.number_input("Error $\sigma_x$:", value=0.5, min_value=0.0, key="ep_x_err")
        y_val = c2.number_input("Value y:", value=5.0, key="ep_y")
        y_err = c2.number_input("Error $\sigma_y$:", value=0.2, min_value=0.0, key="ep_y_err")
        op = st.radio("Operation:", ["Multiplication (x * y)", "Division (x / y)"], horizontal=True, key="ep_op")
        
        if st.button("Calculate Combined Error", key="btn_ep"):

            if x_val == 0 or y_val == 0:
                st.error("Base values cannot be zero for relative error calculations.")
            else:
                f_val = x_val * y_val if "Multiplication" in op else x_val / y_val
                rel_err = math.sqrt((x_err/x_val)**2 + (y_err/y_val)**2)
                abs_err = abs(f_val) * rel_err
                st.success(f"**Final Result:** {f_val:.4f} ± {abs_err:.4f}")


# TAB 6 

with tab6:
    st.subheader(" Advanced Data Formats & Universal SI Converter")
    conv_type = st.selectbox("Select Tool:", [
        "1. Universal SI Converter", "2. CIF: Fractional ↔ Cartesian",
        "3. TEM: Reciprocal Space", "4. SEM: Pixel Calibration", "5. Atomic % to Weight % (Alloy Scaling)"
    ])



    if conv_type == "1. Universal SI Converter":
        st.markdown("**Convert any standard materials science unit.**")
        category = st.radio("Category:", ["Energy", "Pressure", "Length", "Temperature"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        input_val = c1.number_input("Value to Convert", value=1.0, format="%g")

        if category == "Energy":
            units = {"Electron Volts (eV)": 1.60218e-19, "Joules (J)": 1.0, "kcal/mol": 6.9477e-21}
            from_u = c2.selectbox("From Unit", list(units.keys()), index=0)
            to_u = c3.selectbox("To Unit", list(units.keys()), index=1)
            if st.button("Convert Energy"):
                result = input_val * (units[from_u] / units[to_u])
                st.success(f"**Result:** {result:.4e} {to_u.split('(')[-1].replace(')', '')}")

        elif category == "Pressure":
            units = {"Pascals (Pa)": 1.0, "Megapascals (MPa)": 1e6, "Gigapascals (GPa)": 1e9, "psi": 6894.76,
                     "atm": 101325}
            from_u = c2.selectbox("From Unit", list(units.keys()), index=2)
            to_u = c3.selectbox("To Unit", list(units.keys()), index=1)
            if st.button("Convert Pressure"):
                result = input_val * (units[from_u] / units[to_u])
                st.success(f"**Result:** {result:.4g} {to_u.split('(')[-1].replace(')', '')}")

        elif category == "Length":
            units = {"Meters (m)": 1.0, "Millimeters (mm)": 1e-3, "Micrometers (µm)": 1e-6, "Nanometers (nm)": 1e-9,
                     "Angstroms (Å)": 1e-10}
            from_u = c2.selectbox("From Unit", list(units.keys()), index=4)
            to_u = c3.selectbox("To Unit", list(units.keys()), index=3)
            if st.button("Convert Length"):
                result = input_val * (units[from_u] / units[to_u])
                st.success(f"**Result:** {result:.4g} {to_u.split('(')[-1].replace(')', '')}")

        elif category == "Temperature":
            from_u = c2.selectbox("From Unit", ["Celsius (°C)", "Kelvin (K)", "Fahrenheit (°F)"])
            to_u = c3.selectbox("To Unit", ["Celsius (°C)", "Kelvin (K)", "Fahrenheit (°F)"])
            if st.button("Convert Temperature"):
                k_val = input_val + 273.15 if from_u == "Celsius (°C)" else (
                                                                                        input_val - 32) * 5 / 9 + 273.15 if from_u == "Fahrenheit (°F)" else input_val
                result = k_val - 273.15 if to_u == "Celsius (°C)" else (
                                                                                   k_val - 273.15) * 9 / 5 + 32 if to_u == "Fahrenheit (°F)" else k_val
                st.success(f"**Result:** {result:.2f} {to_u.split('(')[-1].replace(')', '')}")

    elif "CIF" in conv_type:
        st.markdown("**Convert Crystal Coordinates to Real Space (assuming cubic)**")
        a_param = st.number_input("Lattice Parameter 'a' (Å)", value=5.431)
        cx, cy, cz = st.columns(3)
        fx = cx.number_input("x (Fractional)", value=0.25)
        fy = cy.number_input("y (Fractional)", value=0.25)
        fz = cz.number_input("z (Fractional)", value=0.25)
        if st.button("Convert to Cartesian (Å)"):
            st.success(f"**X:** {fx * a_param:.4f} Å | **Y:** {fy * a_param:.4f} Å | **Z:** {fz * a_param:.4f} Å")

    elif "TEM" in conv_type:
        st.markdown("**Convert FFT Spots to d-spacing**")
        g_vector = st.number_input("Reciprocal distance 'g' (1/Å)", value=0.492)
        if st.button("Calculate d-spacing", key="calc_d_tab6") and g_vector != 0:
            st.success(f"**Real Space (d):** {1 / g_vector:.4f} Å | **In nm:** {(1 / g_vector) * 0.1:.4f} nm")

    elif "SEM" in conv_type:
        st.markdown("**Convert Image Pixels to Physical Nanometers**")
        scale_nm = st.number_input("Scale Bar Real Length (nm)", value=500.0)
        scale_px = st.number_input("Scale Bar Pixel Length (px)", value=125.0)
        feature_px = st.number_input("Feature Measurement (px)", value=45.0)
        if st.button("Calculate Real Size") and scale_px != 0:
            st.success(
                f"**Calibration:** {scale_nm / scale_px:.2f} nm/px | **Feature Size:** {feature_px * (scale_nm / scale_px):.2f} nm")


    elif "Atomic %" in conv_type:
        st.markdown("**Convert Atomic Percentage (At%) to Weight Percentage (Wt%)**")
        alloy_form = st.text_input("Enter Alloy Formula (e.g., Ti3Al, Fe0.9Ni0.1):", value="Ti3Al")
        if st.button("Calculate Mass Fractions"):
            try:

                clean_form = alloy_form.replace(" ", "").replace(",", "")
                comp = Composition(clean_form)

                c1, c2 = st.columns(2)


                for i, el in enumerate(comp.elements):

                    wt_frac = comp.get_wt_fraction(el)

                    if i % 2 == 0:
                        c1.code(f"{el.symbol}: {wt_frac * 100:.2f} Wt%")
                    else:
                        c2.code(f"{el.symbol}: {wt_frac * 100:.2f} Wt%")

            except Exception as e:
                st.error(f" Error calculating fractions. Details: {e}")

#  TAB 7 

with tab7:
    
    st.markdown("### Synthesis Scaling & Economic Intelligence")
    st.caption("Powered by your Elemental DNA databases (18 Econ, 84 LCA, 43 Safety).")


    c1, c2 = st.columns(2)
    if not st.session_state.batch_data.empty:
        target_mat = c1.selectbox("Select Material for Synthesis:", st.session_state.batch_data["Formula"])
    else:
        target_mat = c1.text_input("Enter Material Formula for Synthesis:", value="LiCoO2")

    target_mass = c2.number_input("Target Batch Mass (Grams):", min_value=0.1, value=10.0, step=1.0)


    smart_translator = {"Graphene": "C", "Graphite": "C", "Water": "H2O"}
    cleaned_mat = smart_translator.get(target_mat, target_mat)


    safe_mat = cleaned_mat.replace(" ", "").replace(",", "")

    try:
        comp = Composition(safe_mat)
        col_mass, col_cost = st.columns(2)

        with col_mass:
            st.markdown("### Mass Breakdown")
            for el in comp.elements:

                wt_fraction = float((comp[el] * el.atomic_mass) / comp.weight)
                st.code(f"Weigh {el.symbol}: {wt_fraction * target_mass:.3f}g")

        with col_cost:
            st.markdown("### Global Intelligence")
            total_cost_per_kg = 0.0

            for el in comp.elements:
                sym = el.symbol
                
                try:
                    dna = get_element_dna(sym)
                    if not isinstance(dna, dict): dna = {}
                except:
                    dna = {}
                
                wt_fraction = float((comp[el] * el.atomic_mass) / comp.weight)

                try:
                    raw_price = str(dna.get("price", "0"))
                    clean_chars = []
                    has_dot = False
                    for char in raw_price:
                        if char.isdigit():
                            clean_chars.append(char)
                        elif char == '.' and not has_dot:
                            clean_chars.append(char)
                            has_dot = True
                    clean_price = "".join(clean_chars)
                    price_val = float(clean_price) if clean_price else 0.0
                except:
                    price_val = 0.0

                total_cost_per_kg += (wt_fraction * price_val)

                with st.expander(f"DNA: {sym}"):
                    st.write(f"**Price:** ${price_val:,.2f}/kg | **Scarcity:** {dna.get('scarcity', 'N/A')}/10")
                    st.write(f"**Note:** {dna.get('issue', 'N/A')}")


            st.divider()
            batch_cost = (total_cost_per_kg / 1000.0) * target_mass
            st.metric("Total Material Cost (Batch)", f"${batch_cost:.2f}")
            st.caption(f"Estimated at ${total_cost_per_kg:,.2f} per Kilogram.")
            
            st.warning(" **Disclaimer:** This is an initial V1 release. Economic data and prices may be inaccurate, outdated, or default to $0.00 if missing from the local database.")

    except Exception as e:
        import traceback
        st.error(f"Engine Error: Could not process '{safe_mat}'.")
        st.code(traceback.format_exc())


    st.divider()
    st.subheader(" Crucible & Hardware Matcher")
    max_temp = st.slider("Max Processing Temperature (°C)", 100, 2500, 1000, 50)

    if max_temp <= 250:
        st.info(" **Teflon (PTFE) / Borosilicate Glass:** Safe for autoclaves & drying.")
    elif max_temp <= 1100:
        st.success(" **Quartz (SiO2):** Standard for CVD/Tube furnaces. Devitrifies >1200°C.")
    elif max_temp <= 1600:

        st.warning(
            " **Alumina (Al2O3) / Platinum (Pt):** Standard high-temp. Keep Pt away from free Silicon or Bismuth (causes low-melting eutectics).")
    elif max_temp <= 2000:
        st.error(
            " **Zirconia (ZrO2) / Molybdenum (Mo):** Extreme heat. Molybdenum requires vacuum or inert atmosphere.")
    else:
        st.error(
            " **Graphite (C) / Tungsten (W):** Vacuum or inert Argon environment absolutely required to prevent incineration.")


    st.divider()
    st.subheader(" Metallurgist's Etchant Directory (ASTM Standards)")
    metal_base = st.selectbox("Select Base Metal / Material:",
                                ["Carbon & Low-Alloy Steel", "Stainless Steel", "Titanium Alloys", "Aluminum Alloys",
                                "Copper & Brass", "Nickel Superalloys", "Silicon / Semiconductors",
                                "Gold & Precious Metals"])

    if metal_base == "Carbon & Low-Alloy Steel":
        st.info(
            "**Nital (2-5%)**: 2-5mL HNO₃ + 100mL Ethanol. Standard for revealing ferrite boundaries and pearlite.")
        st.info("**Picral (4%)**: 4g Picric Acid + 100mL Ethanol. Excellent for carbides and fine pearlite.")
    elif metal_base == "Stainless Steel":
        st.warning(
            "**Vilella's Reagent**: 1g Picric Acid + 5mL HCl + 100mL Ethanol. Best for martensitic and ferritic grades.")

        st.warning(
            "**Glyceregia**: 15mL HNO₃ + 30mL HCl + 45mL Glycerol. Excellent general macro/micro etch for highly alloyed austenitic stainless steels.")
    elif metal_base == "Titanium Alloys":
        st.error(
            "**Kroll's Reagent (Contains HF)**: 2mL HF + 6mL HNO₃ + 92mL Water. Standard for revealing alpha-beta phases. Extreme caution required.")
    elif metal_base == "Aluminum Alloys":
        st.info(
            "**Keller's Reagent**: 2mL HF + 1.5mL HCl + 2.5mL HNO₃ + 95mL Water. Standard microstructural etch for Al-alloys.")
        st.info(
            "**Barker's Reagent**: 5mL HBF₄ + 200mL Water. Used for electrolytic anodizing (view under polarized light).")
    elif metal_base == "Copper & Brass":
        st.info("**Ferric Chloride**: 5g FeCl₃ + 50mL HCl + 100mL Water. General grain boundary etchant.")
        st.info(
            "**Ammonium Hydroxide / Peroxide**: 25mL NH₄OH + 25mL H₂O + 20mL H₂O₂ (3%). Swab fresh. Excellent for brass.")
    elif metal_base == "Nickel Superalloys":
        st.warning(
            "**Marble's Reagent**: 10g CuSO₄ + 50mL HCl + 50mL Water. Great for Inconel and revealing gamma prime (γ').")
        st.warning(
            "**Kalling's No. 2 (Waterless)**: 2g CuCl₂ + 40mL HCl + 40mL Ethanol. Best for Ni-Cu and Ni-Fe-Cr alloys.")
    elif metal_base == "Silicon / Semiconductors":
        st.error(
            "**CP4 Etch**: 3 parts HF + 5 parts HNO₃ + 3 parts CH₃COOH (Acetic). Chemical polishing for Si wafers.")
        st.error(
            "**KOH (Potassium Hydroxide)**: 30% KOH in water at 80°C. Anisotropic etch (etches [100] planes faster than [111]).")
    elif metal_base == "Gold & Precious Metals":
        st.error(
            "**Aqua Regia**: 3 parts HCl + 1 part HNO₃. Must be prepared fresh in a fume hood. Emits highly toxic NO₂ gas.")


#  TAB 8 

with tab8:
    st.markdown("### Spectral Data Refinery")
    st.caption("Savitzky-Golay noise scrubbing and automated peak detection for XRD/Raman/FTIR.")

    uploaded_file = st.file_uploader("Upload Raw Spectral Data (CSV with 'X' and 'Y')", type=["csv"])

    if uploaded_file is not None:
        df_spec = pd.read_csv(uploaded_file)
        if df_spec.shape[1] < 2:
            st.error("CSV must have at least 2 columns (X and Y data).")
            st.stop()
        df_spec = df_spec.iloc[:, :2]   
        df_spec.columns = ['X', 'Y']
    else:
        st.info("No file uploaded. Generating simulated noisy XRD data for demonstration...")
        x_val = np.linspace(20, 80, 500)
        y_base = np.sin(x_val / 5) + np.exp(-(x_val - 40) ** 2 / 2) * 50 + np.exp(-(x_val - 60) ** 2 / 5) * 30
        df_spec = pd.DataFrame({"X": x_val, "Y": y_base + np.random.normal(0, 4, 500)})

    if not df_spec.empty:
        c1, c2 = st.columns(2)
        window = c1.slider("Smoothing Window Size", 5, 51, 15, step=2)

        prominence = c2.number_input("Peak Prominence (Height above baseline)", min_value=0.001, max_value=1000.0, value=0.02, step=0.01)



        safe_window = min(window, len(df_spec) - 1 if len(df_spec) % 2 == 0 else len(df_spec))
        if safe_window <= 3: safe_window = 5 
        
        try:
            df_spec['Smoothed_Y'] = signal.savgol_filter(df_spec['Y'], window_length=safe_window, polyorder=3)
            peaks, _ = signal.find_peaks(df_spec['Smoothed_Y'], prominence=prominence)
        except Exception as e:
            st.error(f"Data is too sparse for Savitzky-Golay smoothing. Need at least {safe_window} data points.")
            df_spec['Smoothed_Y'] = df_spec['Y'] 
            peaks, _ = signal.find_peaks(df_spec['Y'], prominence=prominence)


        fig = px.line(df_spec, x='X', y=['Y', 'Smoothed_Y'], title="Raw vs. Cleaned Spectrum")
        fig.data[0].line.color = 'rgba(255, 255, 255, 0.15)'
        fig.data[1].line.color = '#00E676'
        st.plotly_chart(fig, use_container_width=True)

        if len(peaks) > 0:
            st.success(f"Detected {len(peaks)} major peaks.")
            peak_data = df_spec.iloc[peaks].copy()
            st.dataframe(peak_data[['X', 'Smoothed_Y']].rename(columns={'X': 'Position', 'Smoothed_Y': 'Intensity'}),
                         hide_index=True)
        else:
            st.warning("No prominent peaks found.")


# TAB 9 

with tab9:
    st.markdown("### Mechanical Testing Lab")
    st.caption('DFT-derived approximations')
    
    if st.session_state.active_mat:
        mat_id = st.session_state.active_mat['material_id']
        formula = st.session_state.active_mat['formula']


        bulk, shear, source = fetch_mechanical_tensor(mat_id)


        if bulk is None or shear is None:
            try:
                comp = Composition(formula)
                b_sum, s_sum, weight = 0.0, 0.0, 0.0
                for el in comp.elements:
                    frac = comp.get_atomic_fraction(el)
                    if el.bulk_modulus and el.rigidity_modulus:
                        b_sum += el.bulk_modulus * frac
                        s_sum += el.rigidity_modulus * frac
                        weight += frac
                bulk = b_sum / weight if weight > 0 else 120.0
                shear = s_sum / weight if weight > 0 else 60.0
                source = "Atomic Fraction Rule-of-Mixtures (AI Estimate)"
            except:
                bulk, shear, source = 100.0, 50.0, "System Default"

        c1, c2, c3 = st.columns(3)
        c1.metric("Bulk Modulus (K)", f"{bulk:.1f} GPa", source)
        c2.metric("Shear Modulus (G)", f"{shear:.1f} GPa", source)
        

        if (3 * bulk + shear) == 0:
            youngs = 0.0
        else:
            youngs = (9 * bulk * shear) / (3 * bulk + shear)
        c3.metric("Young's Modulus (E)", f"{youngs:.1f} GPa", "Derived Physics")


        st.divider()
        st.markdown("#### Simulated Destructive Testing")
        mech_tool = st.selectbox("Select Testing Module:",
                                 ["1. Tensile Stress-Strain Analyzer", "2. Universal Hardness Converter"])

        if "Stress-Strain" in mech_tool:
            st.markdown(f"##### Fundamental Elasticity & Ideal Strength for {formula}")
            st.caption("Theoretical limits for a perfect crystal lattice (Frenkel-Orowan bounds).")

            clean_bulk = max(abs(float(bulk)), 1.0)
            clean_shear = max(abs(float(shear)), 1.0)


            youngs = (9 * clean_bulk * clean_shear) / (3 * clean_bulk + clean_shear)
            poisson = (3 * clean_bulk - 2 * clean_shear) / (2 * (3 * clean_bulk + clean_shear))
            poisson = np.clip(poisson, -0.99, 0.499)
            

            e_mpa = youngs * 1000.0
            g_mpa = clean_shear * 1000.0
            
            ideal_tensile = e_mpa / np.pi
            ideal_shear = g_mpa / (2 * np.pi)

            pugh_ratio = clean_bulk / clean_shear
            is_brittle = pugh_ratio < 1.75

            if is_brittle:
                failure_type = "Brittle Cleavage"
                behavior = "Material will violently shatter before undergoing plastic deformation."
            else:
                failure_type = "Ductile Yielding"
                behavior = "Material will plastically deform and bend under high stress."

            st.success(f"**Fracture Mechanics:** {failure_type}")
            st.info(f"**Pugh's Criterion (B/G):** {pugh_ratio:.2f} ({behavior})")

            m1, m2, m3 = st.columns(3)
            m1.metric("Young's Modulus (E)", f"{youngs:.1f} GPa", f"Poisson Ratio: {poisson:.2f}")
            m2.metric("Ideal Tensile Strength", f"{ideal_tensile:,.0f} MPa", "Frenkel-Orowan Limit")
            m3.metric("Ideal Shear Strength", f"{ideal_shear:,.0f} MPa", "Theoretical Maximum")


# HARDNESS CONVERTER
        elif "Hardness" in mech_tool:
            st.markdown("##### Universal Hardness Scale Converter")
            st.caption("Bi-directional conversion between Vickers (HV), Rockwell C (HRC), and Brinell (HBW). Estimates Yield Strength via Tabor's Relation.")
            

            est_hv = int(shear * 10) if shear > 0 else 150
            

            input_scale = st.radio("Select Input Scale:", ["Vickers (HV)", "Rockwell C (HRC)", "Brinell (HBW)"], horizontal=True)
            

            if input_scale == "Vickers (HV)":
                default_val = est_hv
            elif input_scale == "Brinell (HBW)":
                default_val = est_hv * 0.95
            else: # Rockwell C type
                default_val = max(20.0, 116 - (1500 / math.sqrt(max(240, est_hv))))
            
            input_val = st.number_input(f"Enter {input_scale} Value:", min_value=1.0, max_value=5000.0, value=float(default_val), step=1.0)

            if st.button("Calculate Hardness Conversions", use_container_width=True):

                if input_scale == "Vickers (HV)":
                    hv = input_val
                elif input_scale == "Brinell (HBW)":
                    hv = input_val / 0.95
                elif input_scale == "Rockwell C (HRC)":

                    if input_val >= 116: 
                        hv = 5000 
                    else:
                        hv = (1500 / (116 - input_val)) ** 2
                

                hbw = hv * 0.95
                hrc = 116 - (1500 / math.sqrt(hv)) if hv > 0 else 0
                

                est_ys = (hv / 3.0) * 9.807


                hc1, hc2, hc3, hc4 = st.columns(4)
                
                hc1.metric("Vickers (HV)", f"{hv:.0f}")
                

                if hrc < 20:
                    hrc_status = "Too Soft (<20)"
                    hrc_color = "off"
                elif hrc > 68:
                    hrc_status = "Too Hard (>68)"
                    hrc_color = "off"
                else:
                    hrc_status = "Valid Range"
                    hrc_color = "normal"
                hc2.metric("Rockwell C (HRC)", f"{hrc:.1f}", delta=hrc_status, delta_color=hrc_color)

 
                if hbw > 600:
                    hbw_status = "Deforms Ball (>600)"
                    hbw_color = "off"
                else:
                    hbw_status = "Valid Range"
                    hbw_color = "normal"
                hc3.metric("Brinell (HBW)", f"{hbw:.0f}", delta=hbw_status, delta_color=hbw_color)
                    
                hc4.metric("Est. Yield Strength", f"{est_ys:.0f} MPa")




        st.divider()
        st.markdown("####  Advanced Mechanics & Tensor Lab")
        st.caption("Contextual continuum mechanics, fracture analysis, and stress state tensor operations.")

        mech_adv_tool = st.selectbox("Select Advanced Mechanics Module:", [
            "1. Mohr's Circle & Principal Stresses (2D)",
            "2. Hall-Petch Yield Strength Estimator",
            "3. Fracture Toughness & Crack Propagation",
            "4. Steady-State Creep (Norton's Law)"
        ])

        with st.container(border=True):
            if "Mohr" in mech_adv_tool:
                st.markdown("**Calculate Principal Stresses & Max Shear from a 2D Stress State**")
                c1, c2, c3 = st.columns(3)
                sig_x = c1.number_input("Normal Stress X ($\sigma_x$) MPa:", value=100.0)
                sig_y = c2.number_input("Normal Stress Y ($\sigma_y$) MPa:", value=50.0)
                tau_xy = c3.number_input("Shear Stress ($\tau_{xy}$) MPa:", value=30.0)
                
                if st.button("Generate Mohr's Circle"):

                    center = (sig_x + sig_y) / 2.0
                    radius = math.sqrt(((sig_x - sig_y) / 2.0)**2 + tau_xy**2)
                    
                    sig_1 = center + radius
                    sig_2 = center - radius
                    tau_max = radius
                    
                    st.success(f"**Principal Stresses:** $\sigma_1$ = {sig_1:.1f} MPa | $\sigma_2$ = {sig_2:.1f} MPa")
                    st.info(f"**Max Shear Stress ($\tau_{{max}}$):** {tau_max:.1f} MPa | **Center:** {center:.1f} MPa")
                    

                    theta = np.linspace(0, 2*np.pi, 200)
                    circle_x = center + radius * np.cos(theta)
                    circle_y = radius * np.sin(theta)
                    
                    fig_mohr = px.line(x=circle_x, y=circle_y, title="Mohr's Circle of Stress", labels={'x': 'Normal Stress ($\sigma$)', 'y': 'Shear Stress ($\tau$)'})
                    fig_mohr.add_scatter(x=[sig_x, sig_y], y=[-tau_xy, tau_xy], mode='lines+markers', name="Stress Axis", marker=dict(color='red', size=8))
                    fig_mohr.update_traces(line_color='#00E676')
                    fig_mohr.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) # Force 1:1 aspect ratio
                    st.plotly_chart(fig_mohr, use_container_width=True)

            elif "Hall-Petch" in mech_adv_tool:
                st.latex(r"\sigma_y = \sigma_0 + \frac{k_y}{\sqrt{d}}")
                st.markdown("**Calculate yield strength increase due to grain refinement.**")
                c1, c2, c3 = st.columns(3)
                sig_0 = c1.number_input("Friction Stress $\sigma_0$ (MPa):", value=50.0)
                k_y = c2.number_input("Locking Parameter $k_y$ (MPa·m$^{0.5}$):", value=0.5)
                d_um = c3.number_input("Grain Size $d$ (μm):", value=10.0)
                
                if st.button("Calculate Yield Strength", key="calc_ys_tab9"):
                    d_m = d_um * 1e-6
                    sig_y = sig_0 + (k_y / math.sqrt(d_m))
                    st.success(f"**Estimated Yield Strength ($\sigma_y$):** {sig_y:.1f} MPa")

            elif "Fracture" in mech_adv_tool:
                st.latex(r"K_{Ic} = Y \sigma \sqrt{\pi a}")
                st.markdown("**Predict critical crack length or required toughness.**")
                c1, c2, c3 = st.columns(3)
                geom_Y = c1.number_input("Geometry Factor (Y):", value=1.12, help="1.12 for edge crack")
                stress = c2.number_input("Applied Stress $\sigma$ (MPa):", value=250.0)
                crack_mm = c3.number_input("Crack Length $a$ (mm):", value=2.0)
                
                if st.button("Calculate Stress Intensity ($K_I$)"):
                    crack_m = crack_mm * 1e-3
                    k_i = geom_Y * stress * math.sqrt(math.pi * crack_m)
                    st.success(f"**Stress Intensity Factor ($K_I$):** {k_i:.2f} MPa·$\sqrt{{m}}$")
                    st.caption("If $K_I \geq K_{Ic}$ (Fracture Toughness of the material), catastrophic failure occurs.")

            elif "Creep" in mech_adv_tool:
                st.latex(r"\dot{\epsilon} = A \sigma^n \exp\left(-\frac{Q}{RT}\right)")
                st.markdown("**Estimate steady-state creep rate at high temperatures.**")
                c1, c2, c3 = st.columns(3)
                stress = c1.number_input("Applied Stress $\sigma$ (MPa):", value=150.0)
                temp = c2.number_input("Temperature (K):", value=1000.0, min_value=1.0, key="temp_creep_tab9")
                q_act = c3.number_input("Activation Energy $Q$ (kJ/mol):", value=250.0)
                
                c4, c5 = st.columns(2)
                A_const = c4.number_input("Material Constant A:", value=1.5e-4, format="%e")
                n_exp = c5.number_input("Stress Exponent $n$:", value=4.5)
                
                if st.button("Calculate Creep Rate"):
                    r_const = 8.314 # J/mol*K
                    q_joules = q_act * 1000
                    creep_rate = A_const * (stress**n_exp) * math.exp(-q_joules / (r_const * temp))
                    st.success(f"**Steady-State Creep Rate ($\dot{{\epsilon}}$):** {creep_rate:.4e} $s^{{-1}}$")










# TAB 10 

with tab10:
    st.markdown("### Interactive Periodic Table")
    


    
    html_code = """
    <style>
        .ptable { 
            display: grid; 
            grid-template-columns: repeat(18, 1fr); 
            gap: 4px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            padding: 20px;
        }
        .element { 
            border: 1px solid #d3d3d3; 
            border-radius: 6px; 
            padding: 4px; 
            text-align: center; 
            cursor: pointer; 
            transition: all 0.2s ease; 
            box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }
        .element:hover { 
            transform: scale(1.3); 
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); 
            z-index: 100; 
            position: relative;
        }
        .number { font-size: 0.7em; color: #444; text-align: left; font-weight: bold;}
        .symbol { font-size: 1.4em; font-weight: 800; color: #111; margin-top: -2px;}
        .mass { font-size: 0.65em; color: #333; margin-top: 2px;}
    </style>
    <div class="ptable">
    """

    
    for z in range(1, 119):
        try:
            el = Element.from_Z(z)

            
            if 57 <= z <= 71:
                row = 8
                col = z - 57 + 4
            elif 89 <= z <= 103:
                row = 9
                col = z - 89 + 4
            else:
                row = el.row
                col = el.group

            
            bg_color = "#ffffff"
            if el.is_alkali:
                bg_color = "#ffb3ba"
            elif el.is_alkaline:
                bg_color = "#ffdfba"
            elif el.is_transition_metal:
                bg_color = "#ffffba"
            elif el.is_metalloid:
                bg_color = "#baffc9"
            elif el.is_halogen:
                bg_color = "#bae1ff"
            elif el.is_noble_gas:
                bg_color = "#e6ccff"
            elif el.is_lanthanoid:
                bg_color = "#ffb3e6"
            elif el.is_actinoid:
                bg_color = "#d9b3ff"
            elif el.is_post_transition_metal:
                bg_color = "#c2f0c2"

            style = f"grid-column: {col}; grid-row: {row}; background-color: {bg_color};"

           
            hover_text = f"{el.long_name} \\nAtomic No: {z} \\nMass: {el.atomic_mass:.3f} u"

            html_code += f"""
            <div class="element" style="{style}" title="{hover_text}">
                <div class="number">{z}</div>
                <div class="symbol">{el.symbol}</div>
                <div class="mass">{el.atomic_mass:.2f}</div>
            </div>
            """
        except Exception:
            pass  

    html_code += "</div>"

    
    components.html(html_code, height=650, scrolling=False)

    
    st.divider()
    st.markdown("#### Element Classification Guide")

    
    leg1, leg2, leg3, leg4 = st.columns(4)

    with leg1:
        st.markdown("G1 **Alkali Metals** (Highly Reactive)")
        st.markdown("G2 **Alkaline Earth**")
    with leg2:
        st.markdown("G3-G12 **Transition Metals**")
        st.markdown("G13 **Metalloids / Post-Transition**")
    with leg3:
        st.markdown("G17 **Halogens**")
        st.markdown("G18 **Noble Gases** (Inert)")
    with leg4:
        st.markdown(" **Lanthanides** (Rare Earths)")
        st.markdown(" **Actinides** (Radioactive)")

    

    
    st.divider()
    st.markdown("##### Element Overview")
    st.caption(' Some fields may show “N/A” in this Version 1 release.')

    
    active_el = "Fe" 
    if st.session_state.active_mat:
        try:
            active_el = Composition(st.session_state.active_mat['formula']).elements[0].symbol
        except:
            pass

    
    d_search = st.text_input("Target Element (Enter Symbol):", value=active_el).strip().capitalize()

    try:
        if not d_search:
            st.info("  Enter a chemical symbol to begin.")
            st.caption("Use Periodic Table above to see chemical symbols")
        else:
            t_obj = Element(d_search)
            


            scarcity_matrix = {

                "H": 1, "He": 1, "C": 1, "N": 1, "O": 1, "Ne": 1, "Na": 2, "Mg": 2, "Al": 2, "Si": 2, 
                "P": 2, "S": 2, "Cl": 2, "Ar": 1, "K": 2, "Ca": 2, "Ti": 3, "Fe": 2, "Kr": 1, "Xe": 1,

                "V": 4, "Cr": 4, "Mn": 4, "Ni": 4, "Cu": 4, "Zn": 3, "Br": 3, "Rb": 4, "Sr": 4, 
                "Zr": 4, "Ba": 4, "Pb": 4, "Rn": 4,

                "Li": 7, "B": 5, "F": 5, "Sc": 6, "Co": 7, "Ga": 7, "Ge": 7, "As": 5, "Se": 5, "Y": 6, 
                "Nb": 6, "Mo": 5, "Ru": 7, "Ag": 6, "Cd": 5, "In": 7, "Sn": 5, "Sb": 5, "Te": 7, "I": 5, 
                "Cs": 6, "Hf": 6, "Ta": 6, "W": 6, "Re": 7, "Hg": 6, "Tl": 7, "Bi": 5,

                "Be": 8, "Pd": 8, "Os": 8, "Ir": 9, "Pt": 9, "Au": 7, "Rh": 10, "Nd": 9, "Dy": 10, 
                "Pr": 8, "Tb": 9, "Eu": 9, "La": 8, "Ce": 8, "Sm": 8, "Gd": 8, "Ho": 8, "Er": 8, 
                "Tm": 8, "Yb": 8, "Lu": 8, "U": 9, "Th": 8, "Pu": 10
            }
            local_scar = scarcity_matrix.get(d_search, 10) 

            col_left, col_right = st.columns([1, 1.3], gap="large")

            with col_left:
                st.markdown(f"<h1 style='color:#00E676; margin-bottom:0;'>{d_search}</h1>", unsafe_allow_html=True)
                
                try:
                    grp = f"Group {t_obj.group}"
                except ValueError:
                    grp = "F-Block / Rare Earth"
                    
                st.markdown(f"<p style='color:#A0AEC0; font-size:18px;'>{t_obj.long_name} · {grp}</p>", unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Atomic No.", t_obj.Z)
                m2.metric("Mass (u)", f"{t_obj.atomic_mass:.3f}")
                
                with st.container(border=True):
                    st.caption("Supply Chain Scarcity")
                    st.subheader(f"Level {local_scar} / 10")
                    st.progress(int(local_scar) * 10)
                
                st.success("**Tip:** Search any material in the sidebar to automatically sync this dossier.")

            with col_right:
                st.markdown("##### Atomic Properties Radar")
                
                def safe_get(el, attr):
                    try:
                        if attr == "IE1": return el.ionization_energies[0] if el.ionization_energies else None
                        if attr == "Density": return el.density_of_solid if hasattr(el, 'density_of_solid') else None
                        if attr == "X": return float(el.X)
                        val = getattr(el, attr, None)
                        return float(val) if val is not None else None
                    except:
                        return None

                ie1 = safe_get(t_obj, "IE1")
                dens_kg_m3 = safe_get(t_obj, "Density")
                rad = safe_get(t_obj, "atomic_radius")
                tcond = safe_get(t_obj, "thermal_conductivity")
                melt = safe_get(t_obj, "melting_point")
                en = safe_get(t_obj, "X")

                def norm_v(v, max_v):
                    if v is None: return 0
                    return min(100, max(5, (v / max_v) * 100))

                radar_df = pd.DataFrame({
                    "Property": ["Electroneg", "Atomic R", "$IE_1$", "Thermal Cond.", "Density", "Melting"],
                    "Score": [
                        norm_v(en, 4.0), 
                        norm_v(rad, 2.7), 
                        norm_v(ie1, 25.0),
                        norm_v(tcond, 430), 
                        norm_v(dens_kg_m3, 22600), 
                        norm_v(melt, 3800)
                    ]
                })

                fig_rad = px.line_polar(radar_df, r='Score', theta='Property', line_close=True)
                fig_rad.update_traces(fill='toself', line_color='#00E676', fillcolor='rgba(0, 230, 118, 0.2)')
                fig_rad.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0, 105]),
                        angularaxis=dict(gridcolor="#2D3748"),
                        bgcolor="#1A202C" 
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=30, b=30),
                    height=350,
                    font=dict(color="#E0E6ED")
                )
                st.plotly_chart(fig_rad, use_container_width=True)

            st.markdown("##### Measured Analytics Vault")
            
            def fmt(val, unit):
                return f"{val:,.2f} {unit}" if val is not None else "N/A"

            
            st.table(pd.DataFrame({
                "Property": ["Electronegativity", "Atomic Radius", "First Ionization ($IE_1$)", "Density (Solid)", "Thermal Conductivity", "Melting Point"],
                "Measured Value": [fmt(en, "Pauling"), fmt(rad, "Å"), fmt(ie1, "eV"), fmt(dens_kg_m3, "kg/m³"), fmt(tcond, "W/mK"), fmt(melt, "K")],
                "Unit Scale": ["Pauling", "Angstroms", "Electron Volts", "kg/m³", "W/mK", "Kelvin"]
            }))

    except Exception as e:
        st.error(f" Extraction Error for '{d_search}': {str(e)}")
        st.warning("Ensure you enter a valid chemical symbol (e.g., Fe, He, U).")

#  TAB 11

with tab11:
    st.markdown("###   Field Solvers & PDE Laboratory")
    st.caption("Finite Difference Time Domain (FDTD) solvers and non-linear dynamics.")

    sim_choice_11 = st.selectbox("Select Simulator:", [
        "1. Heat Equation Solver (1D Thermal Diffusion)", 
        "2. Non-Linear Dynamics (Lorenz Attractor Chaos)",
        "3. Fourier Series Signal Decomposer",
        "4. Wave Equation Solver (1D String Dynamics)",
        "5. Laplace Electrostatics (2D Potential Grid)"
    ], key="sim_selector_tab11")



    if "Heat Equation" in sim_choice_11:
        st.markdown("##### 1D Heat Conduction (Fourier's Law)")
        st.latex(r"\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}")
        st.caption("Simulates a 100°C heat pulse diffusing through an insulated 1D rod over time.")
        
        c1, c2, c3 = st.columns(3)

        alpha = c1.number_input(r"Thermal Diffusivity ($\alpha$):", value=0.01, min_value=1e-6, format="%g", key="heat_alpha")
        

        t_total = c2.number_input("Total Physical Time (s):", value=2.0, min_value=0.1, key="heat_sim_time")
        grid_points = c3.slider("Spatial Grid Points:", 10, 200, 50, 10, key="heat_grid")

        if st.button("Run FDTD Heat Simulation", use_container_width=True, key="btn_heat"):

            dx = 1.0 / (grid_points - 1)
            max_dt = 0.4 * (dx**2) / alpha 
            
            sim_steps = int(np.ceil(t_total / max_dt))
            if sim_steps < 100: sim_steps = 100 
            
            dt = t_total / sim_steps
            F = alpha * dt / (dx**2) 

            T = np.zeros(grid_points)
            center_idx = grid_points // 2
            pulse_width = max(1, grid_points // 10)
            T[center_idx - pulse_width : center_idx + pulse_width] = 100.0 

            plot_frames = min(sim_steps, 500)
            history = np.zeros((plot_frames, grid_points))
            

            save_indices = np.linspace(0, sim_steps - 1, plot_frames, dtype=int)
            save_idx_pos = 0
            
            with st.spinner(f"Simulating {sim_steps:,} thermodynamic micro-steps..."):
                for t in range(sim_steps):

                    T[1:-1] = T[1:-1] + F * (T[2:] - 2*T[1:-1] + T[:-2])
                    

                    T[0] = T[1]
                    T[-1] = T[-2]
                    

                    if save_idx_pos < plot_frames and t == save_indices[save_idx_pos]:
                        history[save_idx_pos, :] = T
                        save_idx_pos += 1
                        
            x_axis = np.linspace(0, 1.0, grid_points)
            t_axis = np.linspace(0, t_total, plot_frames)
                
            fig = px.imshow(history, x=x_axis, y=t_axis, aspect="auto", origin="lower", 
                            labels=dict(x="Rod Position (m)", y="Physical Time (s)", color="Temp (°C)"),
                            title=rf"Spatiotemporal Thermal Diffusion Map ($\alpha$ = {alpha})", color_continuous_scale="inferno")
            st.plotly_chart(fig, use_container_width=True)

    elif "Lorenz" in sim_choice_11:
        st.markdown("##### Chaos Theory: The Lorenz Attractor")
        st.caption("A system of ordinary differential equations modeling atmospheric convection and the 'Butterfly Effect'.")
        
        c1, c2, c3 = st.columns(3)
        sigma = c1.number_input("Prandtl Number ($\sigma$):", value=10.0, key="lz_sig")
        rho = c2.number_input("Rayleigh Number ($\rho$):", value=28.0, key="lz_rho")
        beta = c3.number_input("Geometric Factor ($\beta$):", value=2.667, key="lz_beta")
        
        if st.button("Simulate Chaotic Dynamics", use_container_width=True, key="btn_lorenz"):
            dt = 0.01
            steps = 5000
            
            xs, ys, zs = np.zeros(steps), np.zeros(steps), np.zeros(steps)
            xs[0], ys[0], zs[0] = 0.0, 1.0, 1.05 # Initial conditions
            
            for i in range(1, steps):
                xs[i] = xs[i-1] + (sigma * (ys[i-1] - xs[i-1])) * dt
                ys[i] = ys[i-1] + (xs[i-1] * (rho - zs[i-1]) - ys[i-1]) * dt
                zs[i] = zs[i-1] + (xs[i-1] * ys[i-1] - beta * zs[i-1]) * dt
                
            fig = px.line_3d(x=xs, y=ys, z=zs, title="3D Phase Space Trajectory")
            fig.update_traces(line=dict(color='#00E676', width=2))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', scene=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig, use_container_width=True)

    elif "Fourier" in sim_choice_11:
        st.markdown("##### Fourier Series Signal Decomposer")
        st.caption("Demonstrates how adding infinite odd sine waves perfectly synthesizes a square wave.")
        terms = st.slider("Number of Harmonics (N):", 1, 50, 5, key="fourier_terms")
        
        if st.button("Synthesize Square Wave", use_container_width=True, key="btn_fourier"):
            x = np.linspace(0, 4*np.pi, 500)
            y = np.zeros_like(x)
            
            for n in range(1, terms*2, 2): 
                y += (4 / (np.pi * n)) * np.sin(n * x)
                
            fig = px.line(x=x, y=y, title=f"Square Wave Approximation (N={terms} terms)")
            fig.update_traces(line_color='#00E676')
            st.plotly_chart(fig, use_container_width=True)

    elif "Wave Equation" in sim_choice_11:
        st.markdown("##### 1D Wave Equation (String Dynamics)")
        st.latex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}")
        st.caption("FDTD simulation of a physical wave bouncing between two fixed boundary walls.")
        
        c1, c2, c3 = st.columns(3)
        wave_speed = c1.number_input("Wave Speed ($c$):", value=1.0, min_value=0.1, max_value=2.0, key="wave_c", help="Capped at 2.0 to maintain Courant (CFL) stability.")
        damping = c2.slider("Friction Damping:", 0.0, 0.05, 0.005, 0.001, key="wave_damp")
        steps = c3.slider("Simulation Time Steps:", 100, 800, 400, 50, key="wave_steps")
        
        if st.button("Simulate Wave Propagation", use_container_width=True, key="btn_wave"):
            nx = 100
            dx = 1.0
            dt = 0.5 
            
            u = np.zeros(nx)
            u_prev = np.zeros(nx)
            

            for i in range(nx):
                u[i] = np.exp(-0.05 * (i - nx//2)**2)
            u_prev[:] = u[:] 
            
            history = np.zeros((steps, nx))
            history[0, :] = u
            
            C2 = (wave_speed * dt / dx)**2
            
            with st.spinner("Calculating temporal grid..."):
                for t in range(1, steps):
                    u_next = np.zeros(nx)
                    u_next[1:-1] = (2*u[1:-1] - u_prev[1:-1]
                                    + C2 * (u[2:] - 2*u[1:-1] + u[:-2])
                                    - damping * (u[1:-1] - u_prev[1:-1]))
                    

                    u_next[0] = 0
                    u_next[-1] = 0
                    
                    u_prev[:] = u[:]
                    u[:] = u_next[:]
                    history[t, :] = u
                
            fig = px.imshow(history, aspect="auto", origin="lower",
                            labels=dict(x="String Position (x)", y="Time (t)", color="Amplitude"),
                            title="Spatiotemporal Wave Heatmap", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

    elif "Laplace" in sim_choice_11:
        st.markdown("##### 2D Electrostatic Potential (Laplace Solver)")
        st.latex(r"\nabla^2 V = 0")
        st.caption("Iteratively solves for the electric field voltage across a 2D grid using the Jacobi method.")
        
        c1, c2 = st.columns(2)
    
        config = c1.radio("Select Charge Configuration:", ["Parallel Plate Capacitor", "Electric Dipole", "Hollow Box (Faraday Cage)"], key="laplace_cfg")
        iters = c2.slider("Relaxation Iterations:", 100, 2000, 800, 100, help="More iterations = higher physical accuracy.", key="laplace_iters")

        
        if st.button("Calculate Electric Potential", use_container_width=True, key="btn_laplace"):
            grid_size = 50
            V = np.zeros((grid_size, grid_size))
            

            if config == "Parallel Plate Capacitor":
                V[10:40, 15] = 100.0  
                V[10:40, 35] = -100.0 
            elif config == "Electric Dipole":
                V[25, 20] = 100.0  
                V[25, 30] = -100.0 
            elif config == "Hollow Box (Faraday Cage)":
                V[10:40, 10] = 100.0
                V[10:40, 40] = 100.0
                V[10, 10:40] = 100.0
                V[40, 10:41] = 100.0
                

            fixed_nodes = V != 0
            
            with st.spinner(f"Running {iters} Jacobi relaxation steps..."):
                for _ in range(iters):
                    V_next = np.copy(V)

                    V_next[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2])

                    V_next[fixed_nodes] = V[fixed_nodes]
                    V = V_next
                    
            fig = px.imshow(V, color_continuous_scale="RdBu_r", title=f"Voltage Contour: {config}", origin='lower')
            fig.update_layout(coloraxis_colorbar=dict(title="Voltage (V)"))
            st.plotly_chart(fig, use_container_width=True)




#  TAB 12

with tab12:
    st.markdown("###  Quantum & Atomistic Laboratory")
    st.caption("Stochastic thermodynamics, quantum state solvers, and statistical mechanics.")

    sim_choice_12 = st.selectbox("Select Simulator:", [
        "1. Monte Carlo Simulator (Ising Model / Ferromagnetism)", 
        "2. Boltzmann Velocity Distribution (Ideal Gas)",
        "3. Kronig-Penney Band Structure (Periodic Lattice)",
        "4. Quantum Wave Packet Propagation (Tunneling)",
        "5. Density of States (3D Semiconductors/Metals)"
    ], key="sim_selector_tab12")



    if "Ising Model" in sim_choice_12:
        st.markdown("##### 2D Ferromagnetic Phase Transition (Metropolis Algorithm)")
        st.caption("Simulates how electron spin alignment breaks down into chaos as temperature increases past the Curie point.")
        
        c1, c2, c3 = st.columns(3)
        grid_size = c1.selectbox("Grid Size (N x N):", [20, 30, 50], index=0, key="ising_grid")
        temp_k = c2.slider("Temperature ($T / T_c$):", 0.1, 5.0, 2.0, 0.1, help="Critical Temp Tc ≈ 2.269", key="ising_temp")
        sweeps = c3.slider("Monte Carlo Sweeps:", 10, 200, 30, 10, key="ising_sweeps")
        
        if st.button("Run Metropolis Monte Carlo", use_container_width=True, key="btn_ising"):

            spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
            
            with st.spinner(f"Running {sweeps} sweeps..."):
                for _ in range(sweeps):
                    for _ in range(grid_size**2):

                        i, j = np.random.randint(0, grid_size, 2)
                        s = spins[i, j]
                        

                        neighbors = spins[(i+1)%grid_size, j] + spins[(i-1)%grid_size, j] + \
                                    spins[i, (j+1)%grid_size] + spins[i, (j-1)%grid_size]
                        
                        delta_E = 2 * s * neighbors
                        

                        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temp_k):
                            spins[i, j] = -s 
                            

            magnetization = np.abs(np.sum(spins)) / (grid_size**2)
            
            fig = px.imshow(spins, color_continuous_scale="RdBu", title=f"Spin Domain Map at T = {temp_k} | Magnetization: {magnetization:.2f}")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    elif "Boltzmann" in sim_choice_12:
        st.markdown("##### Maxwell-Boltzmann Velocity Distribution")
        st.caption("Calculates the statistical probability density of kinetic energy states in an ideal gas.")
        
        c1, c2 = st.columns(2)
        mass_amu = c1.number_input("Particle Mass (amu):", value=4.0, min_value=0.001, help="Default: Helium", key="mb_mass")
        temperature = c2.number_input("Gas Temperature (K):", value=300.0, key="mb_temp", min_value=0.1)
        
        if st.button("Generate Kinetic Distribution", use_container_width=True, key="btn_mb"):
            k_B = 1.380649e-23
            m_kg = mass_amu * 1.660539e-27
            

            v_max = np.sqrt(3 * k_B * temperature / m_kg) * 4
            v = np.linspace(0, v_max, 500)
            

            f_v = 4 * np.pi * (m_kg / (2 * np.pi * k_B * temperature))**1.5 * v**2 * np.exp(-m_kg * v**2 / (2 * k_B * temperature))
            
            v_rms = np.sqrt(3 * k_B * temperature / m_kg)
            v_mp = np.sqrt(2 * k_B * temperature / m_kg)
            
            fig = px.line(x=v, y=f_v, labels={'x': 'Velocity (m/s)', 'y': 'Probability Density'}, title=f"Kinetic Distribution at {temperature}K")
            fig.add_vline(x=v_rms, line_dash="dash", line_color="#F56565", annotation_text=f"V_rms: {int(v_rms)} m/s")
            fig.add_vline(x=v_mp, line_dash="dash", line_color="#00E676", annotation_text=f"V_most_probable: {int(v_mp)} m/s")
            fig.update_traces(line_color='#4299E1', fill='tozeroy', fillcolor='rgba(66, 153, 225, 0.15)')
            st.plotly_chart(fig, use_container_width=True)

    elif "Kronig-Penney" in sim_choice_12:
        st.markdown("##### 1D Kronig-Penney Model (Band Structure)")
        st.latex(r"P \frac{\sin(\alpha a)}{\alpha a} + \cos(\alpha a) = \cos(k a)")
        st.caption("Illustrates the origin of electronic energy bands and bandgaps in periodic crystal lattices.")
        
        P = st.slider("Scattering Power ($P$):", 0.1, 10.0, 3.0, help="Higher P = stronger atomic potential = wider bandgaps.", key="kp_power")
        
        if st.button("Solve Band Structure", use_container_width=True, key="btn_kp"):
            alpha_a = np.linspace(0.01, 4 * np.pi, 1000)
            

            lhs = P * (np.sin(alpha_a) / alpha_a) + np.cos(alpha_a)
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=alpha_a, y=lhs, mode='lines', name='LHS: $P \cdot sinc(\\alpha a) + \cos(\\alpha a)$', line=dict(color='#00E676')))
            

            fig.add_hline(y=1, line_dash="dash", line_color="gray")
            fig.add_hline(y=-1, line_dash="dash", line_color="gray")
            

            allowed = np.abs(lhs) <= 1
            x_allowed = alpha_a[allowed]
            y_allowed = lhs[allowed]
            fig.add_trace(go.Scatter(x=x_allowed, y=y_allowed, mode='markers', name='Allowed Energy Bands', marker=dict(color='#4299E1', size=4)))
            
            fig.update_layout(title="Kronig-Penney Scattering Function", xaxis_title="Energy Parameter ($\alpha a$)", yaxis_title="f($\alpha a$)", yaxis_range=[-3, 3])
            st.plotly_chart(fig, use_container_width=True)

    elif "Wave Packet" in sim_choice_12:
        st.markdown("##### Quantum Wave Packet Propagation (Tunneling)")
        st.caption("Models a Gaussian wave packet dispersing over time and hitting a potential barrier.")
        
        c1, c2 = st.columns(2)
        barrier_height = c1.slider("Barrier Height (Relative):", 0.0, 5.0, 2.0, key="wp_barrier")
        time_evo = c2.slider("Time Evolution Frame:", 0, 50, 10, key="wp_time")
        
        if st.button("Render Wave Packet", use_container_width=True, key="btn_wp"):
            x = np.linspace(-10, 10, 400)
            

            x0 = -5.0 
            p0 = 2.0  
            sigma = 1.0 
            t = time_evo * 0.1
            

            sigma_t = sigma * np.sqrt(1 + (t / sigma**2)**2)
            

            envelope = np.exp(- (x - (x0 + p0*t))**2 / (2 * sigma_t**2)) / np.sqrt(sigma_t)
            

            barrier_x = 0
            transmission = np.exp(-barrier_height)
            reflection = 1 - transmission
            

            psi_prob = np.zeros_like(x)
            for i, pos in enumerate(x):
                if (x0 + p0*t) < barrier_x: 
                    psi_prob[i] = envelope[i]**2
                else: 
                    if pos > barrier_x:
                        psi_prob[i] = (envelope[i]**2) * transmission
                    else:
                       
                        reflected_pos = barrier_x - ((x0 + p0*t) - barrier_x)
                        envelope_ref = np.exp(- (pos - reflected_pos)**2 / (2 * sigma_t**2)) / np.sqrt(sigma_t)
                        psi_prob[i] = (envelope_ref**2) * reflection
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=psi_prob, mode='lines', fill='tozeroy', name='Probability Density $|\Psi|^2$', line_color='#A0AEC0'))
            

            fig.add_shape(type="rect", x0=-0.2, y0=0, x1=0.2, y1=barrier_height*0.1, fillcolor="#F56565", opacity=0.5, line_width=0)
            
            fig.update_layout(title="Wave Packet Dispersion & Tunneling", xaxis_title="Position (x)", yaxis_title="Probability", yaxis_range=[0, 1.2])
            st.plotly_chart(fig, use_container_width=True)

    elif "Density of States" in sim_choice_12:
        st.markdown("##### 3D Density of States ($g(E)$) & Carrier Concentration")
        st.latex(r"g(E) = \frac{m^* \sqrt{2m^* E}}{\pi^2 \hbar^3}")
        st.caption("Visualizes the distribution of available energy states and actual filled electrons (Fermi-Dirac statistics).")
        
        c1, c2 = st.columns(2)
        m_eff = c1.number_input("Effective Mass ($m^*/m_0$):", value=1.0, min_value=0.01, key="dos_mass")
        temp_k = c2.slider("System Temperature (K):", 0, 1000, 300, 50, key="dos_temp")
        fermi_e = st.slider("Fermi Energy $E_F$ (eV):", 0.0, 5.0, 2.5, 0.1, help="Moves the Fermi level through the states.", key="dos_fermi")
        
        if st.button("Calculate State Distribution", use_container_width=True, key="btn_dos"):
            E = np.linspace(0.01, 5.0, 500) # Energy in eV
            

            m_star = m_eff * const.m_e
            E_joules = E * const.e
            

            dos = (m_star * np.sqrt(2 * m_star * E_joules)) / (np.pi**2 * const.hbar**3)
            

            if temp_k == 0:
                f_e = np.where(E <= fermi_e, 1.0, 0.0)
            else:
                kT_eV = (const.k * temp_k) / const.e

                exp_arg = np.clip((E - fermi_e) / kT_eV, -100, 100)
                f_e = 1.0 / (np.exp(exp_arg) + 1.0)
                

            filled_states = dos * f_e
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=E, y=dos, mode='lines', name='Total DOS $g(E)$', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=E, y=filled_states, mode='lines', fill='tozeroy', name='Filled Electron States', line=dict(color='#00E676')))
            
            fig.add_vline(x=fermi_e, line_dash="solid", line_color="#F56565", annotation_text="$E_F$")
            fig.update_layout(title="Carrier Concentration Map", xaxis_title="Energy (eV)", yaxis_title="States / Joules·m³")
            st.plotly_chart(fig, use_container_width=True)

#TAB 13

with tab13:
    st.markdown("###  Advanced Research Utilities")
    st.caption("Data fitting, system stability analysis, scaling laws, and parameter space exploration.")



    import scipy.optimize as opt


    sim_choice_13 = st.selectbox("Select Research Utility:", [
        "1. Nonlinear Curve Fitting & Scaling Detector",
        "2. Eigenmode & Stability Analyzer (2D System)",
        "3. Bifurcation Scanner (Chaos Theory)",
        "4. Parameter Sweep & Phase Map Generator"
    ])



    if "Curve Fitting" in sim_choice_13:
        st.markdown("#####  Nonlinear Curve Fitting Engine")
        st.caption("Automatically fits Linear, Exponential, and Power-Law models to identify underlying physical scaling.")
        
        c1, c2, c3 = st.columns(3)
        noise_level = c1.slider("Experimental Noise Level:", 0.0, 5.0, 1.5)
        true_model = c2.selectbox("Underlying Physical Law:", ["Power Law (e.g. Creep, Fractals)", "Exponential (e.g. Decay, Arrhenius)", "Linear (e.g. Ohmic)"])
        
        if st.button("Generate & Analyze Data", use_container_width=True):

            np.random.seed(42)
            x_data = np.linspace(1, 10, 50)
            
            if "Power" in true_model:
                y_true = 2.5 * (x_data ** 1.8)
            elif "Exponential" in true_model:
                y_true = 5.0 * np.exp(0.4 * x_data)
            else:
                y_true = 10.0 * x_data + 5.0
                
            y_data = y_true + np.random.normal(0, noise_level * np.mean(y_true) * 0.1, len(x_data))
            

            def lin_func(x, a, b): return a * x + b
            def exp_func(x, a, b): return a * np.exp(b * x)
            def pow_func(x, a, b): return a * (x ** b)
            

            def calc_r2(y_actual, y_predicted):
                ss_res = np.sum((y_actual - y_predicted) ** 2)
                ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
                if ss_tot == 0: return 0
                return 1 - (ss_res / ss_tot)


            fits = {}
            try:
                popt_lin, _ = opt.curve_fit(lin_func, x_data, y_data)
                fits['Linear'] = {'y': lin_func(x_data, *popt_lin), 'r2': calc_r2(y_data, lin_func(x_data, *popt_lin))}
            except Exception: pass
                
            try:
                popt_exp, _ = opt.curve_fit(exp_func, x_data, y_data, p0=(1, 0.1), maxfev=5000)
                fits['Exponential'] = {'y': exp_func(x_data, *popt_exp), 'r2': calc_r2(y_data, exp_func(x_data, *popt_exp))}
            except Exception: pass
                
            try:
                popt_pow, _ = opt.curve_fit(pow_func, x_data, y_data, p0=(1, 1), maxfev=5000)
                fits['Power Law'] = {'y': pow_func(x_data, *popt_pow), 'r2': calc_r2(y_data, pow_func(x_data, *popt_pow))}
            except Exception: pass


            if fits:
                best_model = max(fits, key=lambda k: fits[k]['r2'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Raw Experimental Data', marker=dict(color='white', size=8)))
                
                colors = {'Linear': '#A0AEC0', 'Exponential': '#F56565', 'Power Law': '#4299E1'}
                for name, data in fits.items():
                    width = 4 if name == best_model else 2
                    dash = 'solid' if name == best_model else 'dash'
                    fig.add_trace(go.Scatter(x=x_data, y=data['y'], mode='lines', name=f'{name} Fit ($R^2$: {data["r2"]:.3f})', line=dict(color=colors.get(name, '#00E676'), width=width, dash=dash)))
                    
                fig.update_layout(title="Multi-Model Nonlinear Curve Fitting", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f" **Automatic Scaling Detector:** The best mathematical fit is the **{best_model}** model ($R^2$ = {fits[best_model]['r2']:.4f}).")
            else:
                st.error(" All mathematical models failed to converge. The experimental noise is too high or the data is invalid.")


    elif "Eigenmode" in sim_choice_13:
        st.markdown("##### Eigenmode & Stability Analyzer")
        st.caption("Input the Jacobian Matrix elements to determine the dynamic stability of a 2D physical system.")
        st.latex(r"J = \begin{bmatrix} a & b \\ c & d \end{bmatrix}")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            a_val = st.number_input("Element a (Row 1, Col 1):", value=-1.0)
            c_val = st.number_input("Element c (Row 2, Col 1):", value=2.0)
        with c2.container(border=True):
            b_val = st.number_input("Element b (Row 1, Col 2):", value=1.0)
            d_val = st.number_input("Element d (Row 2, Col 2):", value=-1.0)
            
        if st.button("Calculate Eigenvalues & Classify System", use_container_width=True):
            trace = a_val + d_val
            det = (a_val * d_val) - (b_val * c_val)
            discriminant = trace**2 - 4*det
            

            lambda_1 = (trace + np.lib.scimath.sqrt(discriminant)) / 2
            lambda_2 = (trace - np.lib.scimath.sqrt(discriminant)) / 2
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Trace ($Tr$)", f"{trace:.2f}")
            c_res2.metric("Determinant ($\Delta$)", f"{det:.2f}")
            
            st.write(f"**Eigenvalues:** $\lambda_1 =$ `{np.round(lambda_1, 3)}` | $\lambda_2 =$ `{np.round(lambda_2, 3)}`")
            

            if det < 0:
                classification = "  Saddle Point (Unstable, hyperbolic)"
            elif det > 0 and trace > 0:
                if discriminant >= 0: classification = "  Unstable Node (Repeller)"
                else: classification = "  Unstable Spiral (Oscillatory Repeller)"
            elif det > 0 and trace < 0:
                if discriminant >= 0: classification = "  Stable Node (Attractor)"
                else: classification = "  Stable Spiral (Damped Oscillator)"
            elif trace == 0 and det > 0:
                classification = "  Center (Undamped Harmonic Oscillator)"
            else:
                classification = "  Degenerate or Non-Isolated Boundary Case"
                
            c_res3.markdown(f"**System State:**\n\n{classification}")

    elif "Bifurcation" in sim_choice_13:
        st.markdown("##### Bifurcation Scanner (Chaos Theory)")
        st.latex(r"x_{n+1} = r \cdot x_n (1 - x_n)")
        st.caption("Visualizing the transition from deterministic order to complete chaos in the 1D Logistic Map.")
        
        c1, c2 = st.columns(2)
        r_min = c1.number_input("Minimum Parameter (r):", min_value=1.0, max_value=4.0, value=2.5, step=0.1)
        r_max = c2.number_input("Maximum Parameter (r):", min_value=1.0, max_value=4.0, value=4.0, step=0.1)
        
        if st.button("Generate Bifurcation Diagram", use_container_width=True):
            with st.spinner("Calculating chaotic attractors..."):
                R_vals = np.linspace(r_min, r_max, 600)
                iterations = 300
                last = 50 # Keep only the last 50 stable/chaotic orbits
    
                X, Y = [], []
                for r in R_vals:
                    x = 0.5 

                    for _ in range(iterations - last):
                        x = r * x * (1 - x)

                    for _ in range(last):
                        x = r * x * (1 - x)
                        X.append(r)
                        Y.append(x)
                        
                fig_bif = go.Figure(data=go.Scattergl(
                    x=X, y=Y, mode='markers',
                    marker=dict(color='#00E676', size=1.5, opacity=0.3)
                ))
                fig_bif.update_layout(title="Logistic Map Bifurcation", xaxis_title="Growth Rate (r)", yaxis_title="Steady State Population (x)", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bif, use_container_width=True)

    elif "Parameter Sweep" in sim_choice_13:
        st.markdown("##### Parameter Sweep & Phase Map Generator")
        st.caption("Sweeps two independent variables to detect physical phase boundaries or energy minima.")
        st.latex(r"Z(x, y) = \sin(x) \cdot \cos(y) + e^{-\left(\frac{x-5}{2}\right)^2 - \left(\frac{y-5}{2}\right)^2}")
        
        grid_res = st.slider("Sweep Resolution (Grid Density):", 20, 200, 100)
        
        if st.button("Execute 2D Sweep", use_container_width=True):
            x = np.linspace(0, 10, grid_res)
            y = np.linspace(0, 10, grid_res)
            X, Y = np.meshgrid(x, y)
            

            Z = np.sin(X) * np.cos(Y) + np.exp(-((X-5)/2)**2 - ((Y-5)/2)**2) * 2.5
            
            fig_phase = px.imshow(Z, x=x, y=y, color_continuous_scale="Viridis", labels={'x': 'Parameter X ($\alpha$)', 'y': 'Parameter Y ($\beta$)', 'color': 'System Energy (Z)'})
            fig_phase.update_layout(title="2D Parameter Space (Energy Landscape)", template="plotly_dark")
            st.plotly_chart(fig_phase, use_container_width=True)
            st.info("The contour ridges represent system phase boundaries, while the deepest pockets represent stable local minima.")





#TAB 14

with tab14:
    st.markdown("### Quantum Device & Semiconductor Physics")
    st.caption("Advanced mathematical modeling for carrier transport, quantum confinement, and defect kinetics.")





    sim_choice_14 = st.selectbox("Select Device Physics Solver:", [
        "1. Shockley-Read-Hall (SRH) Recombination",
        "2. Fowler-Nordheim (F-N) Tunneling Probability",
        "3. Quantum Well Confinement Energy",
        "4. Drift-Diffusion Current Transport",
        "5. Schottky Thermionic Emission",
        "6. MOSFET C-V & Depletion Profiler",
        "7. Intrinsic Carrier Density (Density of States)"
    ])



    if "Shockley" in sim_choice_14:
        st.markdown("#####   SRH Recombination Kinetics")
        st.latex(r"U_{SRH} = \frac{p n - n_i^2}{\tau_p (n + n_1) + \tau_n (p + p_1)}")
        st.caption("Model how deep-level trap defects act as recombination centers and kill minority carrier lifetimes.")
        
        c1, c2, c3 = st.columns(3)
        n_c = c1.number_input("Electron Density $n$ (cm⁻³):", value=1e15, format="%e")
        p_c = c2.number_input("Hole Density $p$ (cm⁻³):", value=1e10, format="%e")
        n_i = c3.number_input("Intrinsic $n_i$ (cm⁻³):", value=1.0e10, format="%e", help="1e10 is standard for Silicon at 300K")
        
        c4, c5, c6 = st.columns(3)
        tau_n = c4.number_input("Electron Lifetime \\tau_n (s):", value=1e-6, min_value=1e-15, format="%e")
        tau_p = c5.number_input("Hole Lifetime \\tau_p (s):", value=1e-6, min_value=1e-15, format="%e")
        E_t = c6.slider("Trap Level relative to Intrinsic $E_t - E_i$ (eV):", -0.5, 0.5, 0.0)
        
        if st.button("Calculate SRH Rate", use_container_width=True):
            kT = 0.02585 
            n_1 = n_i * math.exp(E_t / kT)
            p_1 = n_i * math.exp(-E_t / kT)
            
            numerator = (p_c * n_c) - (n_i**2)
            denominator = tau_p * (n_c + n_1) + tau_n * (p_c + p_1)
            if denominator == 0:
                st.error("Denominator is zero — all carrier and trap densities cannot simultaneously be zero.")
            else:
                U_srh = numerator / denominator
                st.success(f"**SRH Recombination Rate ($U_{{SRH}}$):** {U_srh:.4e} $cm^{{-3}}s^{{-1}}$")
                if E_t == 0:
                    st.info(" **Physics Note:** Trap at mid-gap — most lethal position for carrier lifetimes.")

    elif "Fowler" in sim_choice_14:
        st.markdown("##### Fowler-Nordheim Tunneling")
        st.latex(r"J_{FN} = A E^2 \exp\left(-\frac{B}{E}\right)")
        st.caption("Quantum tunneling current through a thin dielectric barrier (like a gate oxide) under extreme electric fields.")
        
        c1, c2, c3 = st.columns(3)
        barrier_eV = c1.number_input("Barrier Height $\Phi_B$ (eV):", value=3.2, min_value=0.01, help="Si to SiO2 barrier is ~3.2 eV")
        e_field_MV = c2.number_input("Electric Field (MV/cm):", value=10.0)
        m_eff = c3.number_input("Effective Mass in Oxide ($m^*/m_0$):", value=0.4)
        
        if st.button("Calculate Tunneling Current Density", use_container_width=True):
            if e_field_MV > 0:
                E_field_Vm = e_field_MV * 1e8 
                m_star = m_eff * const.m_e
                q = const.e
                hbar = const.hbar
                

                A = (q**3) / (8 * math.pi * const.h * barrier_eV * q)
                B = (4 * math.sqrt(2 * m_star) * (barrier_eV * q)**1.5) / (3 * q * hbar)
                
                try:
                    J_fn = A * (E_field_Vm**2) * math.exp(-B / E_field_Vm)
                    st.success(f"**Tunneling Current Density ($J_{{FN}}$):** {J_fn:.4e} A/m²  |  {J_fn * 1e-4:.4e} A/cm²")
                except OverflowError:
                    st.error("Field is too weak to cause measurable Fowler-Nordheim tunneling.")
            else:
                st.error("Electric Field must be greater than 0.")

    elif "Quantum Well" in sim_choice_14:
        st.markdown("##### Quantum Confinement Energy")
        st.latex(r"E_n = \frac{\pi^2 \hbar^2 n^2}{2 m^* L^2}")
        st.caption("Calculates the discrete energy levels of an exciton trapped in an infinite 1D potential well (Quantum Dot/Film).")
        
        c1, c2, c3 = st.columns(3)
        L_nm = c1.number_input("Well Width $L$ (nm):", value=5.0)
        m_eff = c2.number_input("Carrier Effective Mass ($m^*/m_0$):", value=0.067, min_value=0.001, help="Default is electron in GaAs")
        n_state = c3.slider("Principal Quantum State ($n$):", 1, 5, 1)
        
        if st.button("Calculate Confinement Shift", use_container_width=True):
            if L_nm > 0:
                L_m = L_nm * 1e-9
                m_star = m_eff * const.m_e
                E_joules = (math.pi**2 * const.hbar**2 * n_state**2) / (2 * m_star * L_m**2)
                E_eV = E_joules / const.e
                st.success(f"**Confinement Energy ($E_{n_state}$):** {E_eV:.4f} eV")
            else:
                st.error("Well width must be greater than 0.")

    elif "Drift-Diffusion" in sim_choice_14:
        st.markdown("##### Drift-Diffusion Current Transport")
        st.latex(r"J_n = q n \mu_n E + q D_n \frac{dn}{dx}")
        st.caption("Calculates the total charge transport combining voltage-driven drift and concentration-driven diffusion.")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Drift Parameters**")
            n_c = st.number_input("Carrier Density $n$ (cm⁻³):", value=1e16, format="%e")
            mu_n = st.number_input("Mobility $\mu$ (cm²/Vs):", value=1350.0, help="Default is Si electron mobility")
            E_field = st.number_input("Electric Field $E$ (V/cm):", value=100.0)
            
        with c2.container(border=True):
            st.markdown("**Diffusion Parameters**")
            grad_n = st.number_input("Gradient $dn/dx$ (cm⁻⁴):", value=1e18, format="%e", help="Rate of change of density over distance")
            temp = st.number_input("Temperature (K):", value=300.0, key="temp_drift_tab14")
            
        if st.button("Calculate Total Current Density", use_container_width=True):
            q = const.e
            V_t = (const.k * temp) / q 
            D_n = mu_n * V_t 
            
            J_drift = q * n_c * mu_n * E_field
            J_diff = q * D_n * grad_n
            
            st.success(f"**Total Current Density ($J_n$):** {J_drift + J_diff:.4f} A/cm²")
            st.info(f"**Drift Component:** {J_drift:.4f} A/cm² | **Diffusion Component:** {J_diff:.4f} A/cm² | **Diffusion Coeff ($D_n$):** {D_n:.2f} cm²/s")

    elif "Schottky" in sim_choice_14:
        st.markdown("##### Schottky Thermionic Emission")
        st.latex(r"J = A^* T^2 \exp\left(-\frac{\Phi_B}{kT}\right) \left[ \exp\left(\frac{qV}{n_{id} kT}\right) - 1 \right]")
        st.caption("Models current leaking over a metal-semiconductor barrier using Richardson's Law.")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            Phi_B = st.number_input("Schottky Barrier Height $\Phi_B$ (eV):", value=0.8)
            m_eff = st.number_input("Effective Mass ($m^*/m_0$):", value=0.3, help="Used to calculate Richardson Constant")
            temp = st.number_input("Temperature $T$ (K):", value=300.0, min_value=1.0)
            
        with c2.container(border=True):
            V_app = st.number_input("Applied Bias $V$ (Volts):", value=0.5, help="Positive = Forward Bias")
            n_id = st.number_input("Ideality Factor ($n_{id}$):", value=1.05, min_value=1.0)
            
        if st.button("Calculate Thermionic Current", use_container_width=True):
            A_star = 120.0 * m_eff 
            kT_eV = (const.k * temp) / const.e
            
            try:
                J_s = A_star * (temp**2) * math.exp(-Phi_B / kT_eV)
                J_total = J_s * (math.exp(V_app / (n_id * kT_eV)) - 1)
                
                st.success(f"**Forward Current Density ($J$):** {J_total:.4e} A/cm²")
                st.caption(f"Reverse Saturation Current ($J_s$): {J_s:.4e} A/cm²")
            except OverflowError:
                st.error("Mathematical Overflow: Voltage is too high for this ideality factor at this temperature.")

    elif "MOSFET" in sim_choice_14:
        st.markdown("##### MOSFET Depletion Profiler")
        st.latex(r"W_{max} = \sqrt{\frac{4 \epsilon_{si} kT \ln(N_a/n_i)}{q^2 N_a}}")
        st.caption("Calculates maximum depletion width and oxide capacitance for a MOS capacitor.")
        
        c1, c2, c3 = st.columns(3)
        N_a = c1.number_input("Substrate Doping $N_a$ (cm⁻³):", value=1e16, format="%e")
        t_ox_nm = c2.number_input("Oxide Thickness $t_{ox}$ (nm):", value=5.0, min_value=0.1)
        temp = c3.number_input("Temperature (K):", value=300.0, key="temp_mos_tab14")
        
        if st.button("Profile MOS Capacitor", use_container_width=True):
            if N_a > 1e10:
                q = const.e
                kT_eV = (const.k * temp) / q
                n_i = 1.0e10 
                eps_0 = 8.854e-14 
                eps_si = 11.7 * eps_0
                eps_ox = 3.9 * eps_0
                

                phi_f = kT_eV * math.log(N_a / n_i)
                

                W_max_cm = math.sqrt((4 * eps_si * phi_f) / (q * N_a))
                W_max_um = W_max_cm * 1e4
                

                C_ox = eps_ox / (t_ox_nm * 1e-7) 

                
                st.success(f"**Max Depletion Width ($W_{{max}}$):** {W_max_um:.4f} μm")
                st.info(f"**Oxide Capacitance ($C_{{ox}}$):** {C_ox * 1e9:.2f} nF/cm² | **Bulk Potential ($\phi_F$):** {phi_f:.3f} V")
            else:
                st.error("Doping $N_a$ must be significantly greater than intrinsic $n_i$ (1e10).")

    elif "Intrinsic Carrier Density" in sim_choice_14:
        st.markdown("##### Intrinsic Carrier Density ($n_i$) via Density of States")
        st.latex(r"n_i = \sqrt{N_c N_v} \exp\left(-\frac{E_g}{2kT}\right)")
        st.caption("Derives exact native carrier concentrations from fundamental effective masses and bandgap.")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            E_g = st.number_input("Bandgap Energy $E_g$ (eV):", value=1.12, help="Default: Silicon")
            temp = st.number_input("System Temperature $T$ (K):", value=300.0, min_value=1.0)
        with c2.container(border=True):
            m_e = st.number_input("Electron DOS Mass ($m_e^*/m_0$):", value=1.08)
            m_h = st.number_input("Hole DOS Mass ($m_h^*/m_0$):", value=0.56)
            
        if st.button("Calculate Native State Density", use_container_width=True):

            Nc = 2.51e19 * (m_e**1.5) * ((temp/300.0)**1.5)
            Nv = 2.51e19 * (m_h**1.5) * ((temp/300.0)**1.5)
            
            kT_eV = (const.k * temp) / const.e
            
            try:
                ni = math.sqrt(Nc * Nv) * math.exp(-E_g / (2 * kT_eV))
                st.success(f"**Intrinsic Carrier Density ($n_i$):** {ni:.4e} cm⁻³")
                st.caption(f"Effective DOS in Conduction Band ($N_c$): {Nc:.2e} cm⁻³ | Valence Band ($N_v$): {Nv:.2e} cm⁻³")
            except OverflowError:
                st.error("Temperature is too close to absolute zero. Intrinsic carriers effectively do not exist.")





#TAB 15

with tab15:
    st.markdown("### Vacuum Kinetics & Thin Film Plasma")
    st.caption("Advanced thermodynamic and kinetic models for PVD, CVD, and ultra-high vacuum (UHV) environments.")





    sim_choice_15 = st.selectbox("Select Vacuum Physics Solver:", [
        "1. Kinetic Mean Free Path & Collision Rate",
        "2. Langmuir Monolayer Formation Time (UHV Contamination)",
        "3. Paschen Curve (Plasma Breakdown Voltage)",
        "4. Hertz-Knudsen Effusion (Deposition Rate)",
        "5. Stoney Equation (Residual Thin-Film Stress)",
        "6. Sigmund Sputter Yield Estimator",
        "7. X-Ray Reflectivity (XRR) Kiessig Fringes"
    ])



    if "Mean Free Path" in sim_choice_15:
        st.markdown("#####  Kinetic Mean Free Path")
        st.latex(r"\lambda = \frac{k_B T}{\sqrt{2} \pi d^2 P}")
        st.caption("Calculates the exact average distance a gas molecule travels before colliding with another molecule.")
        
        c1, c2, c3 = st.columns(3)
        pressure_torr = c1.number_input("Chamber Pressure (Torr):", value=1e-6, format="%e")
        temp_c = c2.number_input("Gas Temperature (°C):", value=25.0)
        diam_nm = c3.number_input("Molecule Diameter (nm):", value=0.37, help="0.37 nm is approx Argon")
        
        if st.button("Calculate Mean Free Path", use_container_width=True):
            if pressure_torr > 0:

                P_pa = pressure_torr * 133.322
                T_k = temp_c + 273.15
                d_m = diam_nm * 1e-9
                
                mfp_m = (const.k * T_k) / (math.sqrt(2) * math.pi * (d_m**2) * P_pa)
                
                st.success(f"**Mean Free Path ($\lambda$):** {mfp_m:.2f} meters")
                if mfp_m > 1.0:
                    st.info(" **Molecular Flow Regime:** Particles will hit the chamber walls before hitting each other.")
                else:
                    st.warning(" **Viscous Flow Regime:** Gas behaves as a continuous fluid (frequent collisions).")
            else:
                st.error("Pressure must be greater than 0.")

    elif "Langmuir" in sim_choice_15:
        st.markdown("##### UHV Monolayer Formation Time")
        st.latex(r"t_{ML} = \frac{N_s}{\Phi \cdot S} \quad \text{where } \Phi = \frac{P}{\sqrt{2 \pi m k T}}")
        st.caption("Calculates how long it takes for a perfectly clean surface to get covered in one atomic layer of background gas.")
        
        c1, c2, c3 = st.columns(3)
        p_torr = c1.number_input("Base Pressure (Torr):", value=1e-8, format="%e")
        mass_amu = c2.number_input("Gas Mass (amu):", value=28.0, help="Default: N2 / CO")
        sticking = c3.number_input("Sticking Coefficient (S):", min_value=0.01, value=1.0, max_value=1.0, help="1.0 means every hitting atom sticks")
        
        if st.button("Calculate Monolayer Time", use_container_width=True):
            if p_torr > 0:
                P_pa = p_torr * 133.322
                T_k = 298.15 
                m_kg = mass_amu * 1.660539e-27
                

                flux = P_pa / math.sqrt(2 * math.pi * m_kg * const.k * T_k)
                

                N_s = 1e19 
                
                t_sec = N_s / (flux * sticking)
                
                st.success(f"**Time to form 1 Monolayer ($t_{{ML}}$):** {t_sec:.2f} seconds")
                if t_sec < 60:
                    st.error(" Contamination is rapid! Ultra-High Vacuum (UHV < 1e-9 Torr) is required to keep this surface clean.")
            else:
                st.error("Pressure must be > 0.")

    elif "Paschen" in sim_choice_15:
        st.markdown("#####   Paschen Curve (Plasma Ignition)")
        st.latex(r"V_B = \frac{B \cdot (p d)}{\ln(A \cdot p d) - \ln\left[\ln\left(1 + \frac{1}{\gamma_{se}}\right)\right]}")
        st.caption("Calculates the exact breakdown voltage required to strike a plasma between two electrodes.")
        
        c1, c2, c3 = st.columns(3)
        gas_type = c1.selectbox("Process Gas:", ["Argon", "Nitrogen", "Air", "Hydrogen"])
        p_torr = c2.number_input("Sputter Pressure (Torr):", value=0.05, min_value=1e-10, format="%e")
        gap_cm = c3.number_input("Electrode Gap $d$ (cm):", value=5.0, min_value=0.001)
        
        if st.button("Calculate Breakdown Voltage", use_container_width=True):

            gas_params = {
                "Argon": {"A": 12.0, "B": 180.0, "gamma": 0.05},
                "Nitrogen": {"A": 12.0, "B": 342.0, "gamma": 0.01},
                "Air": {"A": 15.0, "B": 365.0, "gamma": 0.01},
                "Hydrogen": {"A": 5.0, "B": 130.0, "gamma": 0.01}
            }
            params = gas_params[gas_type]
            

            p_d = p_torr * gap_cm 
            
            try:
                denominator = math.log(params["A"] * p_d) - math.log(math.log(1 + 1/params["gamma"]))
                if denominator <= 0:
                    st.error("Condition falls to the left of the Paschen Minimum. Infinite voltage required (Vacuum insulation).")
                else:
                    V_b = (params["B"] * p_d) / denominator
                    st.success(f"**Breakdown Voltage ($V_B$):** {V_b:.1f} Volts")
            except ValueError:
                st.error("Pressure-Distance product is too small to strike a plasma.")

    elif "Hertz-Knudsen" in sim_choice_15:
        st.markdown("##### Hertz-Knudsen Effusion")
        st.latex(r"\text{Flux } (\Phi) = \frac{P_{vap}}{\sqrt{2 \pi m k T}}")
        st.caption("Models theoretical deposition rate from a Knudsen cell or thermal evaporation boat.")
        
        c1, c2, c3 = st.columns(3)
        p_vap_pa = c1.number_input("Material Vapor Pressure (Pa):", value=1.0, format="%e")
        temp_k = c2.number_input("Crucible Temp (K):", value=1500.0, min_value=1.0)
        mass_amu = c3.number_input("Evaporant Mass (amu):", value=63.5, help="e.g., Cu = 63.5")
        
        if st.button("Calculate Evaporation Rate", use_container_width=True):
            m_kg = mass_amu * 1.660539e-27
            flux_m2_s = p_vap_pa / math.sqrt(2 * math.pi * m_kg * const.k * temp_k)
            

            st.success(f"**Evaporation Flux:** {flux_m2_s:.4e} atoms / (m²·s)")
            st.info("  Flux dictates the theoretical maximum deposition rate before tooling factors and chamber geometry are applied.")

    elif "Stoney" in sim_choice_15:
        st.markdown("##### Stoney Equation (Residual Stress)")
        st.latex(r"\sigma_f = \frac{E_s}{6 (1-\nu_s)} \frac{h_s^2}{h_f R}")
        st.caption("Calculates the immense gigapascals of stress causing a thick wafer to physically bow after deposition.")
        
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            E_s = st.number_input("Substrate Modulus $E_s$ (GPa):", value=130.0, help="Si = 130 GPa")
            nu_s = st.number_input("Poisson's Ratio $\nu_s$:", value=0.28, min_value=0.0, max_value=0.499)

        with c2.container(border=True):
            h_s = st.number_input("Substrate Thick. $h_s$ (μm):", value=500.0)
            h_f = st.number_input("Film Thick. $h_f$ (nm):", value=100.0)
        with c3.container(border=True):
            R_m = st.number_input("Measured Bow Radius $R$ (m):", value=50.0)
            bow_dir = st.radio("Bow Type:", ["Convex (Compressive)", "Concave (Tensile)"])
            
        if st.button("Calculate Thin-Film Stress", use_container_width=True):
            if h_f > 0 and R_m > 0:
                E_pa = E_s * 1e9
                h_s_m = h_s * 1e-6
                h_f_m = h_f * 1e-9
                
                stress_pa = (E_pa / (6 * (1 - nu_s))) * (h_s_m**2 / (h_f_m * R_m))
                stress_mpa = stress_pa / 1e6
                
                sign = "Compressive (Negative)" if "Convex" in bow_dir else "Tensile (Positive)"
                st.success(f"**Residual Film Stress ($\sigma_f$):** {stress_mpa:.1f} MPa ({sign})")
            else:
                st.error("Film thickness and Bow Radius must be > 0.")

    elif "Sigmund" in sim_choice_15:
        st.markdown("##### Sigmund Sputter Yield Estimator")
        st.latex(r"Y \propto \frac{M_i M_t}{(M_i + M_t)^2} \frac{E_{ion}}{U_0}")
        st.caption("Estimates how many target atoms are ejected per incident Argon ion during physical vapor deposition.")
        
        c1, c2, c3 = st.columns(3)
        E_ion = c1.number_input("Ion Energy (eV):", value=500.0)
        M_t = c2.number_input("Target Mass (amu):", value=63.5, help="Cu = 63.5")
        U_0 = c3.number_input("Surface Binding Energy (eV):", value=3.5, min_value=0.01)

        
        if st.button("Estimate Sputter Yield (Y)", use_container_width=True):
            M_i = 39.95 
            alpha = 0.2 if M_t / M_i > 1 else 0.5 
            

            Y = (3 / (4 * math.pi**2)) * alpha * ((4 * M_i * M_t) / (M_i + M_t)**2) * (E_ion / U_0)
            st.success(f"**Estimated Sputter Yield:** {Y:.2f} atoms / ion")

    elif "XRR" in sim_choice_15:
        st.markdown("##### X-Ray Reflectivity (Kiessig Fringes)")
        st.latex(r"t = \frac{\lambda}{2 \Delta \theta}")
        st.caption("Calculates nanoscale thin-film thickness from the interference fringes of an XRR spectrum.")
        
        c1, c2 = st.columns(2)
        wave_nm = c1.number_input("X-Ray Wavelength $\lambda$ (nm):", value=0.15406, help="Cu-Ka")
        delta_theta = c2.number_input("Fringe Spacing $\Delta \theta$ (Degrees):", value=0.45)
        
        if st.button("Calculate Film Thickness", use_container_width=True):
            if delta_theta > 0:
                delta_rad = math.radians(delta_theta)
                thickness = wave_nm / (2 * delta_rad)
                st.success(f"**Estimated Film Thickness ($t$):** {thickness:.2f} nm")
            else:
                st.error("Spacing must be > 0.")

        



#  TAB 16

with tab16:
    st.markdown("### Advanced Solid State & Crystallography")
    st.caption("Reciprocal space generation, tensor stability, and fundamental solid-state electron models.")





    sim_choice_16 = st.selectbox("Select Solid State Module:", [
        "1. Reciprocal Lattice Generator (3D Vectors)",
        "2. Crystal Direction Angle Calculator (Cubic)",
        "3. Born Stability Criteria (Elastic Tensors)",
        "4. Drude Model Conductivity (Electron Gas)",
        "5. Effective Mass from Band Curvature"
    ], key="sim_selector_tab16")



    if "Reciprocal Lattice" in sim_choice_16:
        st.markdown("##### Reciprocal Lattice Vector Generator")
        st.latex(r"\mathbf{b}_1 = 2\pi \frac{\mathbf{a}_2 \times \mathbf{a}_3}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}")
        st.caption("Converts real-space primitive vectors ($a_1, a_2, a_3$) into momentum-space reciprocal vectors ($b_1, b_2, b_3$).")
        
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            st.markdown("**Vector $\mathbf{a}_1$ (Å)**")
            a1_x = st.number_input("x:", value=1.0, key="a1x")
            a1_y = st.number_input("y:", value=0.0, key="a1y")
            a1_z = st.number_input("z:", value=0.0, key="a1z")
        with c2.container(border=True):
            st.markdown("**Vector $\mathbf{a}_2$ (Å)**")
            a2_x = st.number_input("x:", value=0.0, key="a2x")
            a2_y = st.number_input("y:", value=1.0, key="a2y")
            a2_z = st.number_input("z:", value=0.0, key="a2z")
        with c3.container(border=True):
            st.markdown("**Vector $\mathbf{a}_3$ (Å)**")
            a3_x = st.number_input("x:", value=0.0, key="a3x")
            a3_y = st.number_input("y:", value=0.0, key="a3y")
            a3_z = st.number_input("z:", value=1.0, key="a3z")

        if st.button("Generate Reciprocal Lattice", use_container_width=True, key="btn_recip"):
            a1, a2, a3 = np.array([a1_x, a1_y, a1_z]), np.array([a2_x, a2_y, a2_z]), np.array([a3_x, a3_y, a3_z])
            volume = np.dot(a1, np.cross(a2, a3))
            
            if abs(volume) < 1e-8:
                st.error("Vectors are coplanar (Volume = 0). Cannot generate a 3D reciprocal lattice.")
            else:
                b1 = (2 * np.pi * np.cross(a2, a3)) / volume
                b2 = (2 * np.pi * np.cross(a3, a1)) / volume
                b3 = (2 * np.pi * np.cross(a1, a2)) / volume
                
                st.success(f"**Unit Cell Volume:** {volume:.4f} Å³")
                r1, r2, r3 = st.columns(3)
                r1.info(f"**$\mathbf{{b}}_1$:** [{b1[0]:.4f}, {b1[1]:.4f}, {b1[2]:.4f}]")
                r2.info(f"**$\mathbf{{b}}_2$:** [{b2[0]:.4f}, {b2[1]:.4f}, {b2[2]:.4f}]")
                r3.info(f"**$\mathbf{{b}}_3$:** [{b3[0]:.4f}, {b3[1]:.4f}, {b3[2]:.4f}]")

    elif "Direction Angle" in sim_choice_16:
        st.markdown("##### Crystal Direction Angle Calculator (Cubic)")
        st.latex(r"\cos(\theta) = \frac{u_1 u_2 + v_1 v_2 + w_1 w_2}{\sqrt{u_1^2+v_1^2+w_1^2} \sqrt{u_2^2+v_2^2+w_2^2}}")
        st.caption("Calculates the exact angle between two crystallographic directions $[u_1 v_1 w_1]$ and $[u_2 v_2 w_2]$.")
        
        c1, c2 = st.columns(2)
        dir1 = c1.text_input("Direction 1 [u v w]:", value="1 0 0", key="dir1")
        dir2 = c2.text_input("Direction 2 [u v w]:", value="1 1 1", key="dir2")
        
        if st.button("Calculate Angle", use_container_width=True, key="btn_angle"):
            try:
                u1, v1, w1 = map(float, dir1.split())
                u2, v2, w2 = map(float, dir2.split())
                
                dot_prod = (u1*u2 + v1*v2 + w1*w2)
                mag1 = math.sqrt(u1**2 + v1**2 + w1**2)
                mag2 = math.sqrt(u2**2 + v2**2 + w2**2)
                
                if mag1 == 0 or mag2 == 0:
                    st.error("Vectors cannot be [0 0 0] (Zero magnitude).")
                else:
                    cos_theta = dot_prod / (mag1 * mag2)
                    theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))
                    st.success(f"**Angle ($\theta$):** {math.degrees(theta_rad):.2f}° | {theta_rad:.4f} radians")
            except:
                st.error("Please enter space-separated numbers (e.g., '1 1 0').")

    elif "Born Stability" in sim_choice_16:
        st.markdown("##### Born Stability Criteria (Cubic Tensors)")
        st.caption("Evaluates $C_{ij}$ elastic constants to mathematically prove if a cubic crystal structure will collapse under strain.")
        st.latex(r"C_{11} - C_{12} > 0 \quad | \quad C_{11} + 2C_{12} > 0 \quad | \quad C_{44} > 0")
        
        c1, c2, c3 = st.columns(3)
        c11 = c1.number_input("$C_{11}$ (GPa):", value=166.0, help="Default: Silicon", key="born_c11")
        c12 = c2.number_input("$C_{12}$ (GPa):", value=64.0, key="born_c12")
        c44 = c3.number_input("$C_{44}$ (GPa):", value=80.0, key="born_c44")
        
        if st.button("Evaluate Mechanical Stability", use_container_width=True, key="btn_born"):
            cond1 = (c11 - c12) > 0
            cond2 = (c11 + 2 * c12) > 0
            cond3 = c44 > 0
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Condition 1 ($C_{11}-C_{12} > 0$)", "PASS" if cond1 else "FAIL", f"{c11-c12:.1f} GPa", delta_color="normal" if cond1 else "inverse")
            c_res2.metric("Condition 2 ($C_{11}+2C_{12} > 0$)", "PASS" if cond2 else "FAIL", f"{c11+2*c12:.1f} GPa", delta_color="normal" if cond2 else "inverse")
            c_res3.metric("Condition 3 ($C_{44} > 0$)", "PASS" if cond3 else "FAIL", f"{c44:.1f} GPa", delta_color="normal" if cond3 else "inverse")
            
            if cond1 and cond2 and cond3:
                st.success("**Crystal is Mechanically Stable.**")
            else:
                st.error(" **Crystal is Unstable!** This lattice will spontaneously deform or collapse.")

    elif "Drude Model" in sim_choice_16:
        st.markdown("#####   Drude Model Conductivity (Electron Gas)")
        st.latex(r"\sigma = \frac{n e^2 \tau}{m^*}")
        st.caption("Calculates bulk electrical conductivity using the classical free electron gas model.")
        
        c1, c2, c3 = st.columns(3)
        n_c = c1.number_input("Carrier Density $n$ (cm⁻³):", value=8.49e22, format="%e", help="Default: Copper", key="drude_n")
        

        tau = c2.number_input(r"Relaxation Time $\tau$ (fs):", value=25.0, help="Time between collisions", key="drude_tau")
        m_eff = c3.number_input("Effective Mass ($m^*/m_0$):", value=1.0, min_value=0.001, key="drude_m")
        

        if st.button(r"Calculate Conductivity ($\sigma$)", use_container_width=True, key="btn_drude"):
            n_m3 = n_c * 1e6 
            tau_s = tau * 1e-15 
            m_star = m_eff * const.m_e
            
            sigma = (n_m3 * const.e**2 * tau_s) / m_star
            

            resistivity = (1 / sigma) if sigma > 0 else float('inf')
            

            st.success(rf"**Electrical Conductivity ($\sigma$):** {sigma:.4e} S/m")
            st.info(rf"**Electrical Resistivity ($\rho$):** {resistivity:.4e} $\Omega\cdot$m")

    elif "Effective Mass" in sim_choice_16:
        st.markdown("##### Effective Mass from Band Curvature")
        st.latex(r"m^* = \hbar^2 \left[ \frac{\partial^2 E}{\partial k^2} \right]^{-1}")
        st.caption("Estimates the quantum effective mass of a charge carrier based on the parabolic curvature of its energy band.")
        
        curve_ev_a2 = st.number_input("Band Curvature $d^2E/dk^2$ (eV·Å²):", value=3.81, help="Positive = Electron, Negative = Hole", key="eff_curve")
        
        if st.button("Calculate Effective Mass", use_container_width=True, key="btn_effmass"):
            if abs(curve_ev_a2) < 1e-8:
                st.error("Flat band (Zero curvature) results in infinite mass.")
            else:
                
                curvature_J_m2 = curve_ev_a2 * const.e * 1e-20 
                m_star_kg = (const.hbar**2) / curvature_J_m2
                m_relative = m_star_kg / const.m_e
                
                carrier = "Electron (Conduction Band)" if m_relative > 0 else "Hole (Valence Band)"
                st.success(f"**Effective Mass ($m^*$):** {abs(m_relative):.4f} $m_0$")
                st.caption(f"**Carrier Type:** {carrier}")




#  TAB 17

with tab17:
    st.markdown("### Thermodynamics & Transport Phenomena")
    st.caption("Phase equilibria, heat/mass transfer, and statistical mechanics.")


    sim_choice_17 = st.selectbox("Select Thermo/Transport Module:", [
        "1. Gibbs Phase Rule Analyzer",
        "2. Multi-Mode Heat Transfer Solver",
        "3. 1D Diffusion Profile (Fick's Second Law)",
        "4. Statistical Mechanics (Partition Function Z)",
        "5. Thermoelectric Figure of Merit (ZT)"
    ], key="sim_selector_tab17")



    if "Gibbs" in sim_choice_17:
        st.markdown("##### Gibbs Phase Rule Analyzer")
        st.latex(r"F = C - P + N")
        st.caption("Calculates the thermodynamic degrees of freedom (independent intensive properties) of a system.")
        
        c1, c2, c3 = st.columns(3)
        components = c1.number_input("Number of Components ($C$):", min_value=1, value=2, step=1, key="gibbs_c")
        phases = c2.number_input("Number of Phases in Equilibrium ($P$):", min_value=1, value=1, step=1, key="gibbs_p")
        sys_type = c3.radio("System Type ($N$):", ["Standard (T, P variable: N=2)", "Condensed (Solid/Liquid, fixed P: N=1)"], key="gibbs_sys")
        
        if st.button("Calculate Degrees of Freedom", use_container_width=True, key="btn_gibbs"):
            N = 2 if "Standard" in sys_type else 1
            F = components - phases + N
            
            if F < 0:
                st.error(f" **Impossible State (F = {F}).** This combination of phases cannot exist in equilibrium.")
            else:
                st.success(f"**Degrees of Freedom ($F$):** {F}")
                
                if F == 0: st.info(" **Invariant Point:** Temperature and composition are strictly fixed (e.g., Eutectic or Triple point).")
                elif F == 1: st.info(" **Univariant Line:** You can change one variable (e.g., Temp) and the others will automatically adjust to maintain phase equilibrium.")
                elif F == 2: st.info(" **Bivariant Area:** You can independently change two variables (e.g., Temp and Composition) without destroying the phase.")

    elif "Heat Transfer" in sim_choice_17:
        st.markdown("#####  Multi-Mode Heat Transfer Solver")
        st.caption("Calculates steady-state thermal flux ($q$) for conduction, convection, or radiation.")
        
        mode = st.radio("Heat Transfer Mode:", ["Conduction (Fourier)", "Convection (Newton)", "Radiation (Stefan-Boltzmann)"], horizontal=True, key="heat_mode_rad")
        
        c1, c2, c3 = st.columns(3)
        if "Conduction" in mode:
            st.latex(r"q = \frac{k}{L} (T_{hot} - T_{cold})")
            k_cond = c1.number_input("Thermal Cond. $k$ (W/m·K):", value=400.0, help="Copper ≈ 400", key="ht_k")
            L_thick = c2.number_input("Thickness $L$ (m):", value=0.05, key="ht_L")
            dT = c3.number_input("Temp Difference $\Delta T$ (K):", value=50.0, key="ht_dt1")
            if st.button("Calculate Conduction Flux", use_container_width=True, key="btn_cond"):
                if L_thick > 0:
                    flux = (k_cond / L_thick) * dT
                    st.success(f"**Heat Flux ($q$):** {flux:,.1f} W/m²")
                else: st.error("Thickness must be > 0.")
                
        elif "Convection" in mode:
            st.latex(r"q = h (T_{surface} - T_{fluid})")
            h_conv = c1.number_input("Convection Coeff $h$ (W/m²·K):", value=50.0, help="Air natural ≈ 10, Forced ≈ 100", key="ht_h")
            dT = c2.number_input("Temp Difference $\Delta T$ (K):", value=50.0, key="ht_dt2")
            if st.button("Calculate Convection Flux", use_container_width=True, key="btn_conv"):
                flux = h_conv * dT
                st.success(f"**Heat Flux ($q$):** {flux:,.1f} W/m²")
                
        elif "Radiation" in mode:
            st.latex(r"q = \epsilon \sigma (T_{hot}^4 - T_{cold}^4)")
            emissivity = c1.slider("Emissivity $\epsilon$:", 0.0, 1.0, 0.9, key="ht_eps")
            T_hot = c2.number_input("Hot Surface Temp (K):", value=1000.0, key="ht_th")
            T_cold = c3.number_input("Surroundings Temp (K):", value=300.0, key="ht_tc")
            if st.button("Calculate Radiation Flux", use_container_width=True, key="btn_rad"):
                sigma_sb = 5.67e-8 # Stefan-Boltzmann constant
                flux = emissivity * sigma_sb * (T_hot**4 - T_cold**4)
                st.success(f"**Heat Flux ($q$):** {flux:,.1f} W/m²")

    elif "1D Diffusion" in sim_choice_17:
        st.markdown("1D Diffusion Profile (Fick's Second Law)")
        st.latex(r"C(x,t) = C_0 + (C_s - C_0) \left[ 1 - \text{erf}\left(\frac{x}{2\sqrt{Dt}}\right) \right]")
        st.caption("Models time-dependent diffusion into a semi-infinite solid (e.g., carburizing steel, semiconductor doping).")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            C_s = st.number_input("Surface Concentration ($C_s$):", value=1.0, key="diff_cs")
            C_0 = st.number_input("Initial Bulk Concentration ($C_0$):", value=0.0, key="diff_c0")
        with c2.container(border=True):
            D_coeff = st.number_input("Diffusion Coeff $D$ (m²/s):", value=1.0e-11, min_value=1.0e-25, format="%e", key="diff_d")
            time_hrs = st.number_input("Diffusion Time (Hours):", value=10.0, min_value=0.001, key="diff_t")
        
        if st.button("Generate Diffusion Profile", use_container_width=True, key="btn_diff"):
            time_s = time_hrs * 3600
            

            char_length = math.sqrt(D_coeff * time_s)
            depth_m = np.linspace(0, char_length * 4, 200) 
            

            z = depth_m / (2 * math.sqrt(D_coeff * time_s))
            C_xt = C_0 + (C_s - C_0) * (1 - erf(z))     
            
            depth_mm = depth_m * 1000 
            
            fig = px.line(x=depth_mm, y=C_xt, labels={'x': 'Depth from Surface (mm)', 'y': 'Concentration'}, title=f"Concentration Profile at t = {time_hrs} hrs")
            fig.update_traces(line_color='#F56565', fill='tozeroy', fillcolor='rgba(245, 101, 101, 0.15)')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif "Statistical Mechanics" in sim_choice_17:
        st.markdown("##### Statistical Mechanics (2-Level System)")
        st.latex(r"Z = \sum_{i} g_i e^{-E_i / k_B T}")
        st.caption("Calculates the Canonical Partition Function ($Z$) and state probabilities for a simple two-level quantum system.")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Ground State ($E_0$)**")
            g0 = st.number_input("Degeneracy $g_0$:", value=1, min_value=1, step=1, key="stat_g0")
            e0_ev = 0.0  
            st.info("Energy $E_0$: 0.0 eV (Reference)")
            
        with c2.container(border=True):
            st.markdown("**Excited State ($E_1$)**")
            g1 = st.number_input("Degeneracy $g_1$:", value=3, min_value=1, step=1, key="stat_g1")
            e1_ev = st.number_input("Energy $E_1$ (eV):", value=0.05, format="%f", key="stat_e1")
            
        temp_k = st.slider("System Temperature (K):", 1, 1000, 300, key="stat_temp")
        
        if st.button("Evaluate System States", use_container_width=True, key="btn_stat"):
            kT_eV = (const.k * temp_k) / const.e
            

            B0 = g0 * math.exp(-e0_ev / kT_eV)
            B1 = g1 * math.exp(-e1_ev / kT_eV)
            
            Z = B0 + B1
            P0 = B0 / Z
            P1 = B1 / Z
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Partition Function ($Z$)", f"{Z:.4f}")
            c_res2.metric("Prob of Ground State ($P_0$)", f"{P0 * 100:.1f} %")
            c_res3.metric("Prob of Excited State ($P_1$)", f"{P1 * 100:.1f} %")
            
            st.progress(P0, text="Ground State Population")
            st.progress(P1, text="Excited State Population")

    elif "Thermoelectric" in sim_choice_17:
        st.markdown("##### Thermoelectric Figure of Merit ($ZT$)")
        st.latex(r"ZT = \frac{S^2 \sigma}{\kappa_e + \kappa_l} T")
        st.caption("Evaluates the efficiency of a material for converting heat into electricity (Peltier/Seebeck effects).")
        
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Electronic Properties**")
            S_uv = st.number_input("Seebeck Coefficient $S$ ($\mu$V/K):", value=200.0, key="zt_s")
            sigma_sm = st.number_input("Electrical Cond. $\sigma$ (S/cm):", value=1000.0, key="zt_sig")
        with c2.container(border=True):
            st.markdown("**Thermal Properties**")
            k_tot = st.number_input("Total Thermal Cond. $\kappa$ (W/m·K):", value=1.5, key="zt_k")
            temp_k = st.number_input("Operating Temp $T$ (K):", value=300.0, key="zt_t")
            
        if st.button("Calculate ZT", use_container_width=True, key="btn_zt"):

            S_vk = S_uv * 1e-6 # V/K
            sigma_sm_m = sigma_sm * 100 # S/m
            

            PF = (S_vk**2) * sigma_sm_m
            

            if k_tot > 0:
                ZT = (PF * temp_k) / k_tot
                
                st.success(f"**Figure of Merit ($ZT$):** {ZT:.3f}")
                st.info(f"**Power Factor ($S^2\sigma$):** {PF * 1e3:.2f} mW/(m·K²)")
                
                if ZT > 1.0: st.balloons() 
            else:
                st.error("Thermal conductivity must be > 0.")




# TAB 18

with tab18:
    st.markdown("### Applied Mathematics & Tensor Laboratory <span style='color:#F56565; font-size:0.5em; vertical-align:middle; border:1px solid #F56565; padding:2px 6px; border-radius:4px; margin-left:8px;'>BETA</span>", unsafe_allow_html=True)
    st.caption("Matrix algebra, differential equation solvers, and 3D tensor transformations.")

    

    sim_choice_18 = st.selectbox("Select Mathematical Module:", [
        "1. 3D Tensor Rotation Engine (Euler Angles)",
        "2. Linear Equation Solver (3x3 Matrix $Ax=B$)",
        "3. Numerical ODE Solver (Runge-Kutta 4th Order)",
        "4. Numerical Differentiation Engine",
        "5. Crystallographic Symmetry Operations (3x3)"
    ], key="sim_selector_tab18")



    if "Tensor Rotation" in sim_choice_18:
        st.markdown("##### 3D Tensor Rotation Engine")
        st.latex(r"T' = R \cdot T \cdot R^T")
        st.caption("Rotates a symmetric 2nd-rank tensor (e.g., Stress, Strain, Dielectric) using intrinsic Z-Y-X Euler angles.")
        
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1.container(border=True):
            st.markdown("**Original Symmetric Tensor ($T$)**")
            t_xx = st.number_input("T_xx:", value=100.0, key="txx")
            t_yy = st.number_input("T_yy:", value=50.0, key="tyy")
            t_zz = st.number_input("T_zz:", value=-20.0, key="tzz")
            t_xy = st.number_input("T_xy ( = T_yx):", value=30.0, key="txy")
            t_xz = st.number_input("T_xz ( = T_zx):", value=0.0, key="txz")
            t_yz = st.number_input("T_yz ( = T_zy):", value=10.0, key="tyz")
            
        with c3.container(border=True):
            st.markdown("**Euler Rotation Angles (Degrees)**")
            alpha = st.number_input("Z-axis ($\alpha$):", value=45.0, key="ang_a")
            beta = st.number_input("Y-axis ($\beta$):", value=30.0, key="ang_b")
            gamma = st.number_input("X-axis ($\gamma$):", value=0.0, key="ang_c")
            
        if st.button("Calculate Rotated Tensor", use_container_width=True, key="btn_tensor"):
            T = np.array([
                [t_xx, t_xy, t_xz],
                [t_xy, t_yy, t_yz],
                [t_xz, t_yz, t_zz]
            ])
            

            a, b, g = np.radians(alpha), np.radians(beta), np.radians(gamma)
            

            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
            

            R = Rz @ Ry @ Rx
            

            T_prime = R @ T @ R.T
            

            trace = np.trace(T_prime)
            det = np.linalg.det(T_prime)
            vm = np.sqrt(0.5 * ((t_xx-t_yy)**2 + (t_yy-t_zz)**2 + (t_zz-t_xx)**2 + 6*(t_xy**2 + t_yz**2 + t_xz**2)))
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.info("**Rotated Tensor ($T'$):**")
                st.latex(rf"\begin{{bmatrix}} {T_prime[0,0]:.2f} & {T_prime[0,1]:.2f} & {T_prime[0,2]:.2f} \\ {T_prime[1,0]:.2f} & {T_prime[1,1]:.2f} & {T_prime[1,2]:.2f} \\ {T_prime[2,0]:.2f} & {T_prime[2,1]:.2f} & {T_prime[2,2]:.2f} \end{{bmatrix}}")
            with col_res2:
                st.success(f"**Principal Invariants (Unchanged by rotation):**")
                st.write(f"- **Trace ($I_1$):** {trace:.2f}")
                st.write(f"- **Determinant ($I_3$):** {det:.2f}")
                st.write(f"- **Von Mises Equivalent:** {vm:.2f}")

    elif "Linear Equation" in sim_choice_18:
        st.markdown("##### Linear Equation Solver ($Ax = B$)")
        st.caption("Solves a 3-variable system of linear equations (useful for chemical balancing or mass transport constraints).")
        
        st.markdown("**Matrix $A$ (Coefficients)** $\quad \times \quad$ **Vector $x$** $\quad = \quad$ **Vector $B$ (Constants)**")
        
        c1, c2, c3, c4 = st.columns([1,1,1,1])

        a11 = c1.number_input("A11", value=2.0, key="a11")
        a12 = c2.number_input("A12", value=1.0, key="a12")
        a13 = c3.number_input("A13", value=-1.0, key="a13")
        b1 = c4.number_input("B1", value=8.0, key="b1")

        a21 = c1.number_input("A21", value=-3.0, key="a21")
        a22 = c2.number_input("A22", value=-1.0, key="a22")
        a23 = c3.number_input("A23", value=2.0, key="a23")
        b2 = c4.number_input("B2", value=-11.0, key="b2")

        a31 = c1.number_input("A31", value=-2.0, key="a31")
        a32 = c2.number_input("A32", value=1.0, key="a32")
        a33 = c3.number_input("A33", value=2.0, key="a33")
        b3 = c4.number_input("B3", value=-3.0, key="b3")
        
        if st.button("Solve System", use_container_width=True, key="btn_linalg"):
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
            B = np.array([b1, b2, b3])
            
            try:
                x = np.linalg.solve(A, B)
                st.success(" **System Solved Successfully!**")
                rx1, rx2, rx3 = st.columns(3)
                rx1.metric("Variable $x_1$", f"{x[0]:.4f}")
                rx2.metric("Variable $x_2$", f"{x[1]:.4f}")
                rx3.metric("Variable $x_3$", f"{x[2]:.4f}")
            except np.linalg.LinAlgError:
                st.error(" **Singular Matrix:** This system of equations has no unique solution (rows are linearly dependent).")

    elif "ODE Solver" in sim_choice_18:
        st.markdown("##### Numerical ODE Solver (Damped Harmonic Oscillator)")
        st.latex(r"m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + kx = 0")
        st.caption("Uses the Runge-Kutta algorithm (RK45) to solve the 2nd-order differential equation of a mass-spring-damper system.")
        
        c1, c2, c3 = st.columns(3)
        m = c1.number_input("Mass ($m$):", value=1.0, min_value=0.01, key="ode_m")
        c = c2.number_input("Damping ($c$):", value=0.5, min_value=0.0, key="ode_c")
        k = c3.number_input("Spring Constant ($k$):", value=20.0, min_value=0.1, key="ode_k")
        
        c4, c5 = st.columns(2)
        x0 = c4.number_input("Initial Position ($x_0$):", value=1.0, key="ode_x0")
        v0 = c5.number_input("Initial Velocity ($v_0$):", value=0.0, key="ode_v0")
        
        if st.button("Integrate ODE", use_container_width=True, key="btn_ode"):


            def oscillator(t, y):
                x_pos, v_vel = y
                dxdt = v_vel
                dvdt = -(c/m)*v_vel - (k/m)*x_pos
                return [dxdt, dvdt]
                
            t_eval = np.linspace(0, 10, 500)
            solution = solve_ivp(oscillator, [0, 10], [x0, v0], t_eval=t_eval, method='RK45')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=solution.t, y=solution.y[0], mode='lines', name='Position $x(t)$', line=dict(color='#00E676', width=2)))
            fig.add_trace(go.Scatter(x=solution.t, y=solution.y[1], mode='lines', name='Velocity $v(t)$', line=dict(color='#4299E1', width=2, dash='dash')))
            
            fig.update_layout(title="Time-Domain Response", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif "Numerical Differentiation" in sim_choice_18:
        st.markdown("##### Numerical Differentiation Engine")
        st.caption("Calculates local gradients ($f'$ and $f''$) of noisy experimental data using central differences.")
        
        noise_mult = st.slider("Add Synthetic Noise:", 0.0, 1.0, 0.2, key="deriv_noise")
        
        if st.button("Generate & Differentiate Signal", use_container_width=True, key="btn_deriv"):
            x = np.linspace(-5, 5, 200)
            dx = x[1] - x[0]
            

            y_clean = np.tanh(x) + np.sin(2*x)*0.5
            y_noisy = y_clean + np.random.normal(0, noise_mult, len(x))
            

            from scipy.signal import savgol_filter
            y_smooth = savgol_filter(y_noisy, window_length=21, polyorder=3)
            

            dy_dx = np.gradient(y_smooth, dx)
            d2y_dx2 = np.gradient(dy_dx, dx)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_noisy, mode='markers', name='Raw Noisy Data', marker=dict(color='gray', size=4, opacity=0.5)))
            fig.add_trace(go.Scatter(x=x, y=y_smooth, mode='lines', name='$f(x)$ (Smoothed)', line=dict(color='white', width=3)))
            fig.add_trace(go.Scatter(x=x, y=dy_dx, mode='lines', name='$f\'(x)$ (1st Deriv)', line=dict(color='#00E676', width=2)))
            fig.add_trace(go.Scatter(x=x, y=d2y_dx2, mode='lines', name='$f\'\'(x)$ (2nd Deriv)', line=dict(color='#F56565', width=2, dash='dot')))
            
            fig.update_layout(title="Kinematic Gradients", xaxis_title="Independent Variable", yaxis_title="Amplitude", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif "Symmetry Operations" in sim_choice_18:
        st.markdown("##### Crystallographic Symmetry Matrices")
        st.caption("Generates the exact 3x3 transformation matrix for standard point-group symmetry operations.")
        
        sym_op = st.selectbox("Select Symmetry Operation:", [
            "Identity (E)",
            "Inversion Center (i)",
            "Mirror Plane (xy)",
            "Mirror Plane (xz)",
            "Mirror Plane (yz)",
            "2-fold Rotation along Z ($C_2$)",
            "3-fold Rotation along Z ($C_3$)",
            "4-fold Rotation along Z ($C_4$)",
            "6-fold Rotation along Z ($C_6$)"
        ], key="sym_sel")
        
        if st.button("Generate Transformation Matrix", use_container_width=True, key="btn_sym"):
            mat = np.eye(3)
            
            if "Inversion" in sym_op:
                mat = -np.eye(3)
            elif "Mirror Plane (xy)" in sym_op:
                mat[2, 2] = -1
            elif "Mirror Plane (xz)" in sym_op:
                mat[1, 1] = -1
            elif "Mirror Plane (yz)" in sym_op:
                mat[0, 0] = -1
            elif "Rotation" in sym_op:
                angle = 0
                if "2-fold" in sym_op: angle = np.pi
                if "3-fold" in sym_op: angle = 2*np.pi/3
                if "4-fold" in sym_op: angle = np.pi/2
                if "6-fold" in sym_op: angle = np.pi/3
                
                mat[0, 0] = np.cos(angle)
                mat[0, 1] = -np.sin(angle)
                mat[1, 0] = np.sin(angle)
                mat[1, 1] = np.cos(angle)
                

            mat = np.round(mat, 4)
            det = np.linalg.det(mat)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info("**Transformation Matrix ($W$):**")
                st.latex(rf"\begin{{bmatrix}} {mat[0,0]:.2f} & {mat[0,1]:.2f} & {mat[0,2]:.2f} \\ {mat[1,0]:.2f} & {mat[1,1]:.2f} & {mat[1,2]:.2f} \\ {mat[2,0]:.2f} & {mat[2,1]:.2f} & {mat[2,2]:.2f} \end{{bmatrix}}")
            with c2:
                st.success(f"**Determinant:** {det:.0f}")
                if det < 0:
                    st.caption("A negative determinant indicates an operation that changes the chirality (handedness) of the object, such as a reflection or inversion.")
                else:
                    st.caption("A positive determinant indicates a proper rotation that preserves chirality.")



# TAB 19

with tab19:
    st.markdown("### Symbolic Mathematics Engine <span style='color:#F56565; font-size:0.5em; vertical-align:middle; border:1px solid #F56565; padding:2px 6px; border-radius:4px; margin-left:8px;'>BETA</span>", unsafe_allow_html=True)
    st.caption("Algebraic manipulation, analytical calculus, and symbolic systems solving (Powered by SymPy).")
    
    st.info(" **Syntax Rules:** Use Python math syntax. Use `*` for multiplication (e.g., `2*x` not `2x`), `**` for exponents (e.g., `x**2`), and `exp(x)` for $e^x$.")

    sim_choice_19 = st.selectbox("Select Symbolic Module:", [
        "1. Equation Simplifier & Canonical Form",
        "2. Calculus & Auto-Derivation Tool",
        "3. Multi-Equation System Solver",
        "4. Laplace Transform (Analytical)",
        "5. Taylor Series Expansion"
    ], key="sim_selector_tab19")



    if "Simplifier" in sim_choice_19:
        st.markdown("#####   Equation Simplifier")
        st.caption("Expands, factors, and reduces complex algebraic expressions into their simplest canonical form.")
        
        expr_str = st.text_input("Enter Expression:", value="(x**2 + 2*x + 1)/(x + 1)", key="sym_simp_in")
        
        if st.button("Simplify Expression", use_container_width=True, key="btn_sym_simp"):
            try:
                expr = sp.sympify(expr_str)
                simplified = sp.simplify(expr)
                expanded = sp.expand(expr)
                factored = sp.factor(expr)
                
                c1, c2, c3 = st.columns(3)
                with c1.container(border=True):
                    st.markdown("**Fully Simplified:**")
                    st.latex(sp.latex(simplified))
                with c2.container(border=True):
                    st.markdown("**Expanded Form:**")
                    st.latex(sp.latex(expanded))
                with c3.container(border=True):
                    st.markdown("**Factored Form:**")
                    st.latex(sp.latex(factored))
            except sp.SympifyError:
                st.error(" Syntax Error: Please check your mathematical notation (e.g., use `*` for multiplication).")

    elif "Calculus" in sim_choice_19:
        st.markdown("#####  Calculus & Auto-Derivation")
        st.caption("Performs exact analytical derivatives and integrals (no numerical approximation).")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        expr_str = c1.text_input("Enter Function $f(x)$:", value="sin(x) * exp(-2*x)", key="sym_calc_in")
        op = c2.selectbox("Operation:", ["Derivative (d/dx)", "Integral (∫ dx)"], key="sym_calc_op")
        deriv_order = c3.number_input("Order (for derivatives):", min_value=1, max_value=5, value=1, step=1, key="sym_calc_order")
        
        if st.button("Calculate", use_container_width=True, key="btn_sym_calc"):
            try:
                x = sp.Symbol('x')
                expr = sp.sympify(expr_str)
                
                st.markdown("**Original Function $f(x)$:**")
                st.latex(sp.latex(expr))
                
                if "Derivative" in op:
                    res = sp.diff(expr, x, deriv_order)
                    st.markdown(f"**Derivative $f^{{({deriv_order})}}(x)$:**")
                    st.latex(sp.latex(res))
                else:
                    res = sp.integrate(expr, x)
                    st.markdown("**Indefinite Integral $\int f(x) dx$:**")
                    st.latex(sp.latex(res) + " + C")
            except Exception as e:
                st.error(f" Calculation Error: Check your notation. Details: {str(e)}")

    elif "Multi-Equation" in sim_choice_19:
        st.markdown("#####  Multi-Equation System Solver")
        st.caption("Solves systems of non-linear or algebraic equations symbolically.")
        
        eqs_str = st.text_area("Enter Equations (comma separated). Assume '= 0' for each:", value="x**2 + y**2 - 25, x - y - 1", key="sym_sys_eqs")
        vars_str = st.text_input("Variables to solve for (comma separated):", value="x, y", key="sym_sys_vars")
        
        if st.button("Solve System", use_container_width=True, key="btn_sym_sys"):
            try:
                eqs = [sp.sympify(eq.strip()) for eq in eqs_str.split(',')]
                vars_list = [sp.Symbol(v.strip()) for v in vars_str.split(',')]
                
                st.markdown("**System of Equations:**")
                for eq in eqs:
                    st.latex(sp.latex(sp.Eq(eq, 0)))
                
                solutions = sp.solve(eqs, vars_list, dict=True)
                
                if solutions:
                    st.success(f" Found {len(solutions)} analytical solution set(s):")
                    for i, sol in enumerate(solutions):
                        st.markdown(f"**Solution {i+1}:**")
                        sol_latex = ", \quad ".join([f"{sp.latex(k)} = {sp.latex(v)}" for k, v in sol.items()])
                        st.latex(sol_latex)
                else:
                    st.warning("No analytical solutions found or the system is inconsistent.")
            except Exception as e:
                st.error(f" Syntax Error: {str(e)}")

    elif "Laplace Transform" in sim_choice_19:
        st.markdown("#####  Laplace Transform Tool")
        st.caption("Converts time-domain functions $f(t)$ into s-domain frequencies $F(s)$, critical for control theory.")
        
        c1, c2 = st.columns(2)
        dir_op = c1.radio("Transform Direction:", ["Laplace: $f(t) \\rightarrow F(s)$", "Inverse Laplace: $F(s) \\rightarrow f(t)$"], key="sym_lap_dir")
        expr_str = c2.text_input("Enter Function:", value="exp(-a*t) * sin(w*t)" if "Inverse" not in dir_op else "w / ((s + a)**2 + w**2)", key="sym_lap_in")
        
        if st.button("Compute Transform", use_container_width=True, key="btn_sym_lap"):
            try:
                t, s = sp.symbols('t s')

                a, w = sp.symbols('a w', real=True, positive=True)
                expr = sp.sympify(expr_str, locals={'t': t, 's': s, 'a': a, 'w': w})
                
                if "Inverse" not in dir_op:
                    F_s, _, _ = sp.laplace_transform(expr, t, s)
                    st.markdown("**Laplace Transform $F(s)$:**")
                    st.latex(sp.latex(F_s))
                else:
                    f_t = sp.inverse_laplace_transform(expr, s, t)
                    st.markdown("**Inverse Laplace Transform $f(t)$:**")
                    st.latex(sp.latex(f_t))
            except Exception as e:
                st.error(f" Transform Error: This specific transform may not have a closed-form analytical solution, or syntax is incorrect. ({str(e)})")

    elif "Taylor Series" in sim_choice_19:
        st.markdown("#####  Taylor Series Expansion")
        st.caption("Approximates any complex function as a polynomial around a specific point.")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        expr_str = c1.text_input("Enter Function $f(x)$:", value="sin(x)", key="sym_taylor_in")
        point = c2.number_input("Expand around point (a):", value=0.0, key="sym_taylor_pt")
        terms = c3.number_input("Number of Terms (Order):", min_value=1, max_value=15, value=5, step=1, key="sym_taylor_terms")
        
        if st.button("Generate Series Expansion", use_container_width=True, key="btn_sym_taylor"):
            try:
                x = sp.Symbol('x')
                expr = sp.sympify(expr_str)
                series = sp.series(expr, x, x0=point, n=terms)
                
                st.markdown(f"**Taylor Series (up to order {terms}):**")
                st.latex(sp.latex(series))
            except Exception as e:
                st.error(" Expansion Error: Check notation.")

# TAB 20

with tab20:
    st.markdown("### Advanced Engineering & Systems Analysis <span style='color:#F56565; font-size:0.5em; vertical-align:middle; border:1px solid #F56565; padding:2px 6px; border-radius:4px; margin-left:8px;'>BETA</span>", unsafe_allow_html=True)

    st.caption("Note: Dimensionless π groups are not unique. Your result may appear as the reciprocal or a power of textbook forms (e.g., 1/Re instead of Re).\n Any mathematically independent dimensionless group is valid.")

    sim_choice_20 = st.selectbox("Select Engineering Module:", [
        "1. Buckingham π Theorem (Dimensional Analysis)",
        "2. Analytical Sensitivity & Perturbation Ranker",
        "3. Inverse Problem Solver (Goal Seek)"
    ], key="sim_selector_tab20")


    if "Buckingham" in sim_choice_20:
        st.markdown("##### Buckingham $\\pi$ Engine (Dimensional Analysis)")
        st.caption("Automatically derives fundamental dimensionless groups (like Reynolds or Froude numbers) from a list of variables.")
        

        st.info(r" **Instructions:** Enter your variables and their dimensions as exponents of Mass [M], Length [L], Time [T], and Temperature [\Theta].")
        
        c1, c2 = st.columns([1, 3])
        num_vars = c1.number_input("Number of Variables:", min_value=3, max_value=8, value=4, step=1, key="pi_num")
        
        with c2.container():
            var_data = []
            cols = st.columns(5)
            cols[0].markdown("**Variable**")
            cols[1].markdown("**[M]**")
            cols[2].markdown("**[L]**")
            cols[3].markdown("**[T]**")
            cols[4].markdown(r"**[\Theta]**") 
            
            defaults = [
                ("F", 1, 1, -2, 0),    
                ("rho", 1, -3, 0, 0),  
                ("v", 0, 1, -1, 0),    
                ("A", 0, 2, 0, 0),    
                ("mu", 1, -1, -1, 0),  
                ("g", 0, 1, -2, 0),    
                ("L", 0, 1, 0, 0),     
                ("T", 0, 0, 0, 1)      
            ]
            
            for i in range(num_vars):
                def_v = defaults[i] if i < len(defaults) else (f"v{i+1}", 0, 0, 0, 0)
                cc = st.columns(5)
                v_name = cc[0].text_input(f"Name {i+1}", value=def_v[0], key=f"pi_n_{i}", label_visibility="collapsed")
                m_dim = cc[1].number_input(f"M {i+1}", value=def_v[1], key=f"pi_m_{i}", label_visibility="collapsed")
                l_dim = cc[2].number_input(f"L {i+1}", value=def_v[2], key=f"pi_l_{i}", label_visibility="collapsed")
                t_dim = cc[3].number_input(f"T {i+1}", value=def_v[3], key=f"pi_t_{i}", label_visibility="collapsed")
                th_dim = cc[4].number_input(f"Th {i+1}", value=def_v[4], key=f"pi_th_{i}", label_visibility="collapsed")
                var_data.append({"name": v_name, "dims": [m_dim, l_dim, t_dim, th_dim]})

        if st.button("Derive $\\pi$ Groups", use_container_width=True, key="btn_pi"):
            try:
                dim_matrix = []
                var_names = []
                for v in var_data:
                    dim_matrix.append(v["dims"])
                    var_names.append(sp.Symbol(v["name"]))
                
                A = sp.Matrix(dim_matrix).T
                null_space = A.nullspace()
                
                if not null_space:
                    st.warning("No dimensionless groups can be formed. The variables are dimensionally independent.")
                else:
                    k = len(var_names) - A.rank()
                    st.success(rf"**Buckingham $\pi$ Theorem:** $n$ ({len(var_names)}) variables minus $k$ ({A.rank()}) fundamental dimensions = **{k} Independent $\pi$ Groups**.")
                    
                    for idx, vec in enumerate(null_space):
                        lcm_val = 1
                        for val in vec:
                            denom = int(sp.fraction(val)[1])
                            lcm_val = abs(lcm_val * denom) // math.gcd(lcm_val, denom)
                            
                        int_vec = vec * lcm_val
                        
                        pi_expr = 1
                        for i, exp in enumerate(int_vec):
                            if exp != 0:

                                pi_expr *= var_names[i] ** int(exp)
                        

                        st.markdown(rf"**$\pi_{{{idx+1}}}$ Group:**")
                        st.latex(rf"\pi_{{{idx+1}}} = " + sp.latex(pi_expr))
            except Exception as e:
                st.error(f" Matrix Error: {str(e)}")

    elif "Sensitivity" in sim_choice_20:
        st.markdown("##### Analytical Sensitivity & Ranker")

        st.caption("Impact factors use logarithmic sensitivity")
        
        c1, c2 = st.columns([1, 1])
        expr_str = c1.text_input("System Equation $f(x, y, ...)$:", value="x**2 * y / sqrt(z)", key="sens_eq")
        vars_input = c2.text_input("Define Nominal Values (comma separated):", value="x=5, y=10, z=4", key="sens_vals")
        
        if st.button("Analyze System Sensitivity", use_container_width=True, key="btn_sens"):
            try:
                expr = sp.sympify(expr_str)
                val_dict = {}
                
                pairs = [p.strip() for p in vars_input.split(',') if p.strip()]
                for pair in pairs:
                    if '=' not in pair:
                        st.error("Syntax Error: Ensure values are formatted like 'x=5, y=10'")
                        st.stop()
                    

                    parts = pair.split('=')
                    k, v = parts[0], parts[1]
                    val_dict[sp.Symbol(k.strip())] = float(v.strip())
                
                baseline_expr = expr.evalf(subs=val_dict)
                

                if not baseline_expr.is_number:
                    st.error("Evaluation Error: Ensure all variables in the equation are defined in Nominal Values.")
                    st.stop()
                if not baseline_expr.is_real:
                    st.error("Math Domain Error: Result is complex (e.g., square root of negative number).")
                    st.stop()
                    
                baseline = float(baseline_expr)
                
                sensitivities = []
                for var, val in val_dict.items():
                    partial_deriv = sp.diff(expr, var)
                    deriv_val = float(partial_deriv.evalf(subs=val_dict))
                    
                    normalized_sens = (deriv_val * val) / baseline if baseline != 0 else 0
                    
                    sensitivities.append({
                        "Variable": str(var),
                        "Nominal Value": val,
                        "Partial Deriv (Absolute)": deriv_val,
                        "Impact Factor (Normalized)": abs(normalized_sens)
                    })
                
                df_sens = pd.DataFrame(sensitivities).sort_values(by="Impact Factor (Normalized)", ascending=True)
                
                c_res1, c_res2 = st.columns([1, 1.5])
                with c_res1:
                    st.success(f"**Baseline Output:** {baseline:.4f}")
                    st.dataframe(df_sens[["Variable", "Impact Factor (Normalized)"]].sort_values(by="Impact Factor (Normalized)", ascending=False), hide_index=True)
                with c_res2:
                    fig = px.bar(df_sens, x="Impact Factor (Normalized)", y="Variable", orientation='h', title="Relative Parameter Influence")
                    fig.update_traces(marker_color='#00E676')
                    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f" Parse Error: Ensure equation notation is correct and variables match. ({str(e)})")

    elif "Inverse" in sim_choice_20:
        st.markdown("##### Inverse Problem Solver (Goal Seek)")
        st.caption("Numerically solves an equation backwards to find the required input for a target output.")
        
        st.info(" Example: If equation is `50*exp(-E_a / (8.314 * 300))`, solve for `E_a` to get a target rate.")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        eq_str = c1.text_input("Equation (with 1 unknown variable):", value="50 * exp(-E_a / (8.314 * 300))", key="inv_eq")
        target_val = c2.number_input("Target Output ($y$):", value=0.05, format="%g", key="inv_target")
        guess = c3.number_input("Initial Guess for Variable:", value=10000.0, format="%g", key="inv_guess")
        
        if st.button("Seek Target", use_container_width=True, key="btn_inv"):
            try:
                expr = sp.sympify(eq_str)
                
                free_syms = list(expr.free_symbols)
                if len(free_syms) != 1:
                    st.error(f"Equation must contain exactly ONE unknown variable. Found: {len(free_syms)}")
                    st.stop()
                    
                target_var = free_syms[0]
                f_num = sp.lambdify(target_var, expr - target_val, 'numpy')
                x1_val = guess * 1.1 if guess != 0 else 0.1
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sol = opt.root_scalar(f_num, x0=guess, x1=x1_val, method='secant', maxiter=100)
                
                if sol.converged:
                    residual = abs(f_num(sol.root))
                    
                    is_valid = False
                    if not np.isnan(residual) and not np.isinf(residual):
                        if residual < 1e-3: 
                            is_valid = True
                        elif target_val != 0 and (residual / abs(target_val)) < 1e-3:
                            is_valid = True

                    if is_valid:
                        st.success(f" **Goal Achieved!** Required input value: **{target_var} = {sol.root:.4e}**")
                        st.caption(f"Algorithm converged in {sol.iterations} iterations.")
                    else:
                        st.error(f" **False Convergence Detected:** The algorithm halted, but the result is mathematically incorrect (Residual Error: {residual:.2e}). Try a closer initial guess to prevent mathematical divergence.")
                else:
                    st.error(" Algorithm failed to converge. Try a different initial guess.")
            except Exception as e:
                st.error(f" Computation Error: Check syntax or ensure the target is mathematically possible. ({str(e)})")




