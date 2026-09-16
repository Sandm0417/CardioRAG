"""Tests for CardioRAG normalization module."""

import pytest

from cardiorag.normalize import (
    detect_emergency,
    detect_out_of_bounds,
    extract_patient_factors,
    normalize_input,
    normalize_terms,
)


class TestNormalize:
    """Test input normalization pipeline."""

    def test_extract_age(self):
        factors = extract_patient_factors("患者，年龄65岁，确诊冠心病3年")
        assert factors.get("age") == 65

    def test_extract_lvef(self):
        factors = extract_patient_factors("LVEF 35%，NYHA II级")
        assert factors.get("LVEF") == "35%"

    def test_extract_chads_vasc(self):
        factors = extract_patient_factors("CHA2DS2-VASc评分3分，需抗凝")
        assert factors.get("CHA2DS2_VASc") == 3

    def test_extract_crcl(self):
        factors = extract_patient_factors("eGFR 45ml/min")
        assert factors.get("CrCl_or_eGFR") == 45

    def test_extract_ntprobnp(self):
        factors = extract_patient_factors("NT-proBNP 1200 pg/mL")
        assert factors.get("NT_proBNP") == 1200

    def test_extract_ctni(self):
        factors = extract_patient_factors("cTnI 0.15 ng/mL")
        assert factors.get("cTnI") == 0.15

    def test_detect_emergency_chest_pain(self):
        is_emergency, keywords = detect_emergency("我胸痛伴有冷汗，持续30分钟")
        assert is_emergency is True
        assert len(keywords) > 0

    def test_detect_emergency_syncope(self):
        is_emergency, _ = detect_emergency("刚才突然晕厥了")
        assert is_emergency is True

    def test_detect_emergency_en(self):
        is_emergency, _ = detect_emergency("sudden dyspnea with chest pain")
        assert is_emergency is True

    def test_no_emergency_normal(self):
        is_emergency, _ = detect_emergency("血压140/90，需要吃药吗？")
        assert is_emergency is False

    def test_detect_oob_dose(self):
        is_oob, patterns = detect_out_of_bounds("你应该将阿司匹林增加至200mg")
        assert is_oob is True

    def test_detect_oob_stop_med(self):
        is_oob, _ = detect_out_of_bounds("建议你停用氯吡格雷")
        assert is_oob is True

    def test_detect_oob_surgery(self):
        is_oob, _ = detect_out_of_bounds("你需要做PCI支架手术")
        assert is_oob is True

    def test_no_oob_normal(self):
        is_oob, _ = detect_out_of_bounds("根据指南，阿司匹林常用于冠心病二级预防")
        assert is_oob is False

    def test_normalize_input_full(self):
        result = normalize_input("患者年龄65岁男性，左心室射血分数LVEF 35%，脑钠肽NT-proBNP 800，最近胸痛加重了")
        assert result["patient_factors"]["age"] == 65
        assert result["patient_factors"]["LVEF"] == "35%"
        assert result["patient_factors"]["NT_proBNP"] == 800
        assert result["is_emergency"] is True  # "胸痛加重"
        assert result["language"] == "zh"
