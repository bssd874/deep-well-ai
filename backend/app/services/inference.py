import pandas as pd
import numpy as np
import pickle
import os
import re
import threading
from typing import Dict, List, Tuple

class DeepWellInference:
    def __init__(self, model_dir: str = "app/models"):
        self.model_dir = model_dir
        self.eor_dir = os.path.join(model_dir, "EOR")
        self.lithology_dir = os.path.join(model_dir, "Lithology")
        self.risk_dir = os.path.join(model_dir, "Risk")
        
        self.eor_cache = {}
        self.lithology_cache = {}
        self.risk_cache = {}
        
        self._lock = threading.Lock()

        # Define Physical Limits (Sanity Gates)
        # Format: 'column': (min_val, max_val, action)
        # action: 'clip' (potong ke limit), 'null' (jadikan NaN), 'ignore'
        self.physics_limits = {
            'gr': (0, 300, 'clip'),        # Gamma Ray apapun di atas 300 biasanya noise
            'nphi': (0, 1.0, 'clip'),      # Porosity desimal. 
            'rhob': (1.0, 3.5, 'null'),    # Density di bawah 1 (air) atau di atas 3.5 (jenis mineral berat langka) perlu dicek
            'rdep': (0.01, 10000, 'null'), # Resistivity
            'pef': (0, 15, 'clip')         # Photoelectric factor
        }

        self.available_versions = self._scan_all_versions()
        self.official_models = self._build_model_registry()
        
        self._load_model("eor", self.eor_dir, self.available_versions['eor'][-1] if self.available_versions['eor'] else None)
        self._load_model("lithology", self.lithology_dir, self.available_versions['lithology'][-1] if self.available_versions['lithology'] else None)
        self._load_model("risk", self.risk_dir, self.available_versions['risk'][-1] if self.available_versions['risk'] else None)

    # ... (scan & registry functions tetap sama) ...
    def _scan_all_versions(self):
        return {
            "eor": self._scan_dir(self.eor_dir, "EOR"),
            "lithology": self._scan_dir(self.lithology_dir, "Lithology"),
            "risk": self._scan_dir(self.risk_dir, "Risk")
        }

    def _scan_dir(self, directory, prefix):
        versions = set()
        if not os.path.exists(directory): return []
        for filename in os.listdir(directory):
            match = re.match(rf"{prefix}_(v\d+)\.pkl", filename, re.IGNORECASE)
            if match: versions.add(match.group(1).lower())
        return sorted(list(versions), key=lambda x: int(x[1:]))

    def _build_model_registry(self):
        models = []
        eor_vers = self.available_versions['eor']
        for ver in eor_vers:
            is_latest = (ver == eor_vers[-1])
            models.append({
                "id": f"deepwell-unified-{ver}",
                "object": "model",
                "owned_by": "deepwell-team",
                "status": "active" if is_latest else "legacy"
            })
        if not models:
            models.append({"id": "deepwell-unified-logic", "object": "model", "owned_by": "deepwell-team", "status": "active"})
        return models

    def get_available_models(self):
        active = f"deepwell-unified-{self.available_versions['eor'][-1]}" if self.available_versions['eor'] else "deepwell-unified-logic"
        return {"models": self.official_models, "active_model": active}
    
    def is_valid_model(self, model_id: str): return any(m["id"] == model_id for m in self.official_models)

    def _load_model(self, model_type, directory, version):
        if not version: return
        cache = getattr(self, f"{model_type}_cache")
        if version in cache: return

        with self._lock:
            if version in cache: return
            
            prefix = model_type.capitalize()
            path = os.path.join(directory, f"{prefix}_{version.upper()}.pkl")
            
            if not os.path.exists(path):
                cache[version] = {"type": "hardcoded"}
                return

            try:
                with open(os.path.join(directory, f"{prefix}_{version.upper()}_preprocessor.pkl"), 'rb') as f: preprocessor = pickle.load(f)
                with open(os.path.join(directory, f"{prefix}_{version.upper()}_y_enc.pkl"), 'rb') as f: encoder = pickle.load(f)
                with open(path, 'rb') as f: model = pickle.load(f)
                
                cache[version] = {"type": "ml", "model": model, "preprocessor": preprocessor, "encoder": encoder}
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
                cache[version] = {"type": "hardcoded"}

    def _sanitize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Deep Data Sanitization:
        1. Standardize Null values (-999.25, -9999).
        2. Handle Infinity.
        3. Auto-normalize units (e.g., percentage to decimal).
        """
        warnings = []
        
        # 1. Standard Nulls in Industry
        df.replace([-999.25, -9999, -999], np.nan, inplace=True)
        
        # 2. Infinity handling
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 3. Unit Auto-Normalization (NPHI check)
        if 'nphi' in df.columns:
            # If NPHI values > 1, assume it's percentage and convert to decimal
            if df['nphi'].max() > 1.0:
                df['nphi'] = df['nphi'] / 100.0
                warnings.append("INFO: NPHI values detected > 1.0. Auto-converted from percentage to decimal.")

        return df, warnings

    def _validate_physics(self, df: pd.DataFrame) -> List[str]:
        """
        Physical Limit & Sanity Gates.
        Checks if sensor data is physically possible.
        """
        warnings = []
        
        for col, (min_val, max_val, action) in self.physics_limits.items():
            if col in df.columns:
                # Check violations
                mask_low = df[col] < min_val
                mask_high = df[col] > max_val
                
                if mask_low.any() or mask_high.any():
                    count = mask_low.sum() + mask_high.sum()
                    warnings.append(f"WARNING: {count} rows violate physical limits for {col.upper()} ({min_val}-{max_val}).")
                    
                    if action == 'clip':
                        df[col].clip(min_val, max_val, inplace=True)
                    elif action == 'null':
                        df[col].where(~(mask_low | mask_high), np.nan, inplace=True)
                        
        return warnings

    def _determine_intent(self, columns: List[str]) -> Tuple[bool, bool]:
        cols = set(c.lower() for c in columns)
        has_eor = 'formation' in cols or 'gravity_api' in cols
        has_lithology = 'gr' in cols or 'rhob' in cols
        return has_eor, has_lithology

    async def predict_unified(self, df: pd.DataFrame, model_id: str):
        warnings = []
        
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]

        # Aliasing
        if 'rdep' not in df.columns and 'rmed' in df.columns:
            df.rename(columns={'rmed': 'rdep'}, inplace=True)
        
        # 1. Sanitize Data (Deep Clean)
        df, sanitize_warnings = self._sanitize_data(df)
        warnings.extend(sanitize_warnings)

        # 2. Validate Physics (Sanity Gates)
        physics_warnings = self._validate_physics(df)
        warnings.extend(physics_warnings)

        # Check if data is empty after cleaning
        if df.empty:
             return {"model_used": model_id, "warnings": ["ERROR: Data empty after sanitization."], "predictions": []}

        eor_status, lithology_status = self._determine_intent(df.columns)

        # ... (Version resolution logic same as before) ...
        eor_ver = model_id.split("-")[-1] if model_id.startswith("deepwell-unified-") else None
        lith_ver = self.available_versions['lithology'][-1] if self.available_versions['lithology'] else "hardcoded"
        risk_ver = self.available_versions['risk'][-1] if self.available_versions['risk'] else "hardcoded"

        # Load models
        if eor_status:
            self._load_model("eor", self.eor_dir, eor_ver)
            eor_artifacts = self.eor_cache.get(eor_ver)
            if eor_artifacts["type"] == "hardcoded": warnings.append("WARNING: EOR output is MOCK data.")
        
        if lithology_status:
            self._load_model("lithology", self.lithology_dir, lith_ver)
            lith_artifacts = self.lithology_cache.get(lith_ver)
            if lith_artifacts["type"] == "hardcoded": warnings.append("WARNING: Lithology output is MOCK data.")

            self._load_model("risk", self.risk_dir, risk_ver)
            risk_artifacts = self.risk_cache.get(risk_ver)
            if risk_artifacts["type"] == "hardcoded": warnings.append("WARNING: Risk output is MOCK data.")

        # Batch Processing
        results = pd.DataFrame()
        
        if 'depth' in df.columns: results['depth'] = df['depth']
        elif 'depth_ft_m21' in df.columns: results['depth'] = df['depth_ft_m21']

        if eor_status:
            eor_res = self._run_inference_batch("eor", df, eor_artifacts)
            results = pd.concat([results, eor_res], axis=1)
        
        if lithology_status:
            lith_res = self._run_inference_batch("lithology", df, lith_artifacts)
            risk_res = self._run_inference_batch("risk", df, risk_artifacts)
            results = pd.concat([results, lith_res, risk_res], axis=1)

        return {
            "model_used": model_id,
            "detection_type": "eor" if eor_status else "lithology",
            "warnings": warnings,
            "predictions": results.to_dict(orient='records')
        }

    def _run_inference_batch(self, model_type, df: pd.DataFrame, artifacts: dict):
        if not artifacts: return pd.DataFrame()
        
        if artifacts["type"] == "hardcoded":
            return self._hardcoded_logic_batch(model_type, df)
        
        try:
            # ML Logic with Feature Reindexing (Critical for Scikit-Learn)
            proc = artifacts["preprocessor"]
            mdl = artifacts["model"]
            enc = artifacts["encoder"]
            
            # Get expected features from preprocessor if available
            # Most sklearn preprocessors have get_feature_names_out() or feature_names_in_
            expected_cols = None
            if hasattr(proc, 'feature_names_in_'):
                expected_cols = proc.feature_names_in_
            elif hasattr(proc, 'get_feature_names_out'):
                expected_cols = proc.get_feature_names_out()
            
            input_df = df.copy()
            
            # Reindex to match training data structure (fills missing with NaN/0, ignores extra)
            if expected_cols is not None:
                input_df = df.reindex(columns=expected_cols, fill_value=np.nan)
            
            processed = proc.transform(input_df)
            raw = mdl.predict(processed)
            labels = enc.inverse_transform(raw)
            
            return pd.DataFrame({f"{model_type}_prediction": labels})
        except Exception as e:
            return pd.DataFrame({f"{model_type}_prediction": [f"Model Error: {str(e)}"] * len(df)})

    def _hardcoded_logic_batch(self, model_type, df: pd.DataFrame):
        # MOCK DATA: Strict validation logic.
        
        if model_type == "eor":
            return pd.DataFrame({"eor_strategy": ["Mock EOR Data"] * len(df)})

        if model_type == "lithology":
            required = ['gr', 'pef']
            # Check if columns exist AND are not all NaN
            missing = [c for c in required if c not in df.columns or df[c].isna().all()]
            
            if missing:
                msg = f"INPUT ERROR: Missing critical columns for Lithology: {missing}"
                return pd.DataFrame({"lithology": [msg] * len(df)})

            gr = df['gr'].fillna(0) # Fill NaNs just for calculation to avoid crash
            pef = df['pef'].fillna(0)
            
            conditions = [
                (gr < 50) & (pef < 3.0),
                gr > 80,
                pef > 4.0
            ]
            choices = ['Sandstone', 'Shale', 'Dolomite']
            lith = np.select(conditions, choices, default='Limestone')
            return pd.DataFrame({"lithology": lith})

        if model_type == "risk":
            required = ['rdep', 'nphi']
            missing = [c for c in required if c not in df.columns or df[c].isna().all()]
            
            if missing:
                msg = f"INPUT ERROR: Missing critical columns for Risk: {missing}"
                return pd.DataFrame({
                    "risk_score": [None] * len(df),
                    "confidence": [0.0] * len(df),
                    "anomaly_detected": [False] * len(df),
                    "risk_error": [msg] * len(df)
                })

            rdep = df['rdep'].fillna(df['rdep'].mean()) # Impute mean for calculation stability
            nphi = df['nphi'].fillna(df['nphi'].mean())
            
            base_risk = np.where((rdep < 10) & (nphi > 0.25), 0.8, 0.2)
            noise = np.random.normal(0, 0.05, len(df))
            risk = np.clip(base_risk + noise, 0, 1)
            
            return pd.DataFrame({
                "risk_score": np.round(risk, 2),
                "confidence": np.round(np.random.uniform(0.85, 0.95, len(df)), 2),
                "anomaly_detected": risk > 0.7
            })
            
        return pd.DataFrame()