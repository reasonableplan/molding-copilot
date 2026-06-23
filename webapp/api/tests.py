"""api 서비스 레이어 테스트. DB 미접근 → SimpleTestCase."""
from django.test import SimpleTestCase

from . import service


class ServiceTests(SimpleTestCase):
    def test_list_shots_returns_requested_category(self):
        shots = service.list_shots("가스")
        self.assertTrue(shots)
        self.assertTrue(all(s["groundTruth"] == "가스" for s in shots))

    def test_diagnose_returns_camelcase_contract(self):
        idx = service.list_shots("미성형")[0]["idx"]
        d = service.diagnose(idx, "supervised")
        for key in ("isAnomaly", "pValue", "alpha", "gateMode", "zbars", "groundTruth"):
            self.assertIn(key, d)
        self.assertIsInstance(d["isAnomaly"], bool)
        self.assertIsInstance(d["zbars"], list)

    def test_supervised_gate_detects_a_real_defect(self):
        # 지도 게이트(recall 89%)는 대표 미성형 결함을 잡아야 한다.
        idx = service.list_shots("미성형")[0]["idx"]
        self.assertTrue(service.diagnose(idx, "supervised")["isAnomaly"])

    def test_trust_metrics_camelcase(self):
        t = service.trust_metrics()
        self.assertEqual(t["recallSup"], 89)
        self.assertIn("cwDiffCi", t)
