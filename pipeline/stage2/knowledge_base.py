# pipeline/stage2/knowledge_base.py
# Loads and provides lookup methods for the Clinical Knowledge Base (knowledge.json).

import json
import re
from pathlib import Path


class KnowledgeBase:
    def __init__(self, path: str):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    # ── Clinical Parameters ──────────────────────────────────────────────────

    def get_param(self, name: str) -> dict | None:
        """Return the clinical parameter dict by name (case-insensitive, alias-aware)."""
        params = self._data.get("clinical_parameters", {})
        # Direct lookup
        if name in params:
            return params[name]
        # Case-insensitive
        name_lower = name.lower().replace(" ", "_")
        for key, val in params.items():
            if key.lower().replace(" ", "_") == name_lower:
                return val
        # Alias search
        _aliases = {
            "spo2": "SpO2",
            "oxygen saturation": "SpO2",
            "heart_rate": "heart_rate",
            "heart rate": "heart_rate",
            "blood_pressure_systolic": "blood_pressure_systolic",
            "blood pressure": "blood_pressure_systolic",
            "systolic": "blood_pressure_systolic",
            "diastolic": "blood_pressure_diastolic",
            "glucose": "glucose",
            "respiratory_rate": "respiratory_rate",
            "respiratory rate": "respiratory_rate",
            "temperature": "temperature",
        }
        canonical = _aliases.get(name.lower())
        if canonical and canonical in params:
            return params[canonical]
        return None

    def get_all_param_names(self) -> list:
        return list(self._data.get("clinical_parameters", {}).keys())

    # ── User Roles ───────────────────────────────────────────────────────────

    def get_all_roles(self) -> list:
        return list(self._data.get("user_roles", {}).keys())

    def get_authorized_roles(self) -> list:
        return [
            role for role, attrs in self._data.get("user_roles", {}).items()
            if attrs.get("authorized", False)
        ]

    def get_unauthorized_roles(self) -> list:
        return [
            role for role, attrs in self._data.get("user_roles", {}).items()
            if not attrs.get("authorized", False)
        ]

    def get_role(self, name: str) -> dict | None:
        roles = self._data.get("user_roles", {})
        if name in roles:
            return roles[name]
        name_lower = name.lower().replace(" ", "_")
        for key, val in roles.items():
            if key.lower().replace(" ", "_") == name_lower:
                return val
        return None

    def expand_abstract_role(self, abstract_value: str) -> list:
        """
        Given an abstract role description, return concrete role names.
        e.g. "unauthorized user roles" → ["lab_technician", "patient"]
        """
        av_lower = abstract_value.lower()
        if "unauthorized" in av_lower:
            return self.get_unauthorized_roles()
        if "authorized" in av_lower:
            return self.get_authorized_roles()
        if "all" in av_lower or "any" in av_lower:
            return self.get_all_roles()
        # Default: return all roles
        return self.get_all_roles()

    # ── Workflow States ──────────────────────────────────────────────────────

    def get_workflow(self, name: str) -> dict | None:
        workflows = self._data.get("workflow_states", {})
        if name in workflows:
            return workflows[name]
        name_lower = name.lower().replace(" ", "_")
        for key, val in workflows.items():
            if key.lower().replace(" ", "_") == name_lower:
                return val
        return None

    def get_all_workflow_names(self) -> list:
        return list(self._data.get("workflow_states", {}).keys())

    # ── Alert Types ──────────────────────────────────────────────────────────

    def get_alert_types(self) -> list:
        return self._data.get("alert_types", [])

    # ── Time Thresholds ──────────────────────────────────────────────────────

    def get_time_threshold(self, name: str) -> int | None:
        thresholds = self._data.get("time_thresholds", {})
        if name in thresholds:
            return thresholds[name]
        name_lower = name.lower().replace(" ", "_")
        for key, val in thresholds.items():
            if key.lower().replace(" ", "_") == name_lower:
                return val
        return None

    # ── Domain Knowledge Update ──────────────────────────────────────────────

    def update_from_domain_knowledge(self, sentences: list) -> None:
        """
        Parse DOMAIN_KNOWLEDGE sentences and attempt to update KB values.
        e.g. "SpO2 below 88% is critical" → updates critical_low for SpO2
        """
        param_aliases = {
            "spo2": "SpO2",
            "heart rate": "heart_rate",
            "blood pressure": "blood_pressure_systolic",
            "glucose": "glucose",
            "respiratory rate": "respiratory_rate",
            "temperature": "temperature",
        }

        pattern = re.compile(
            r'(SpO2|heart rate|blood pressure|glucose|respiratory rate|temperature)'
            r'.+?(below|above|exceeds|greater than|less than|over|under)\s+(\d+\.?\d*)',
            re.IGNORECASE,
        )

        for sentence in sentences:
            for m in pattern.finditer(sentence):
                param_raw = m.group(1).lower()
                operator = m.group(2).lower()
                value_str = m.group(3)

                canonical_param = param_aliases.get(param_raw)
                if not canonical_param:
                    continue

                params = self._data.get("clinical_parameters", {})
                if canonical_param not in params:
                    continue

                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Heuristic: if the sentence contains "critical" and "below" → update critical_low
                sent_lower = sentence.lower()
                if "critical" in sent_lower and operator in ("below", "less than", "under"):
                    params[canonical_param]["critical_low"] = value
                elif "critical" in sent_lower and operator in ("above", "exceeds", "greater than", "over"):
                    params[canonical_param]["critical_high"] = value
                elif "normal" in sent_lower and operator in ("below", "less than"):
                    params[canonical_param]["normal_low"] = value
                elif "normal" in sent_lower and operator in ("above", "exceeds", "greater than"):
                    params[canonical_param]["normal_high"] = value
