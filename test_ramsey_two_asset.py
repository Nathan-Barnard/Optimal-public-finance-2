import tempfile
import unittest
from pathlib import Path

import numpy as np

import ramsey_two_asset as model


class RamseyTwoAssetTests(unittest.TestCase):
    def test_accounting_uses_risky_and_safe_income_tax_base(self) -> None:
        cal = model.Calibration(gamma=1.2)
        k = 2.0
        L = 0.4
        H = 0.5
        tau = 0.2
        T = 0.1
        omega = 0.08
        _, _, r_k, _ = model._production_block(cal, 1, k)
        r_f = model._safe_rate(cal, 1, k, L, H, tau)
        _, kdot, Ldot = model._node_flow_and_drift(cal, 1, k, L, tau, H, T, omega)
        Y, w, _, _ = model._production_block(cal, 1, k)
        W = model._owner_wealth(k, L)
        expected_kdot = Y - (w + T) - omega * W - (cal.delta + cal.g) * k
        B = L + H
        expected_Ldot = r_f * B + T - H * r_k - tau * ((k - H) * r_k + r_f * B)
        self.assertAlmostEqual(kdot, expected_kdot)
        self.assertAlmostEqual(Ldot, expected_Ldot)
        self.assertAlmostEqual(W, k + L)
        self.assertAlmostEqual(model._private_risky_share(k, L, H), (k - H) / (k + L))

    def test_baseline_smoke_run(self) -> None:
        cal = model.Calibration(
            k_points=9,
            L_points=9,
            tau_points=5,
            H_points=5,
            transfer_points=5,
            max_outer=3,
            max_inner=3,
            peel_steps=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            spec = model.ExperimentSpec(
                output_dir=tmp,
                run_baseline=True,
                run_comparative_statics=False,
                run_log_benchmark=False,
                plot_2d_heatmaps=False,
                plot_slices=False,
                verbose=False,
            )
            results = model.run_experiment(cal, spec)
        baseline = results["baseline"]
        self.assertIsNotNone(baseline)
        self.assertGreater(len(baseline["history"]), 0)
        self.assertGreater(int(baseline["M1"].sum()), 0)
        self.assertGreater(int(baseline["M0"].sum()), 0)
        self.assertTrue(np.all(baseline["omega1"][baseline["M1"]] > 0.0))
        self.assertTrue(np.all(baseline["omega0"][baseline["M0"]] > 0.0))

    def test_log_benchmark_reports_small_fixed_point_gap(self) -> None:
        cal = model.Calibration(
            gamma=1.0,
            k_points=7,
            L_points=7,
            tau_points=5,
            H_points=5,
            transfer_points=5,
            max_outer=8,
            max_inner=8,
            peel_steps=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            spec = model.ExperimentSpec(
                output_dir=tmp,
                run_baseline=False,
                run_comparative_statics=False,
                run_log_benchmark=True,
                plot_2d_heatmaps=False,
                plot_slices=False,
                verbose=False,
            )
            results = model.run_experiment(cal, spec)
        bench = results["log_benchmark"]["benchmark"]
        self.assertLessEqual(bench["omega1_max_abs_diff"], 5e-4)
        self.assertLessEqual(bench["omega0_max_abs_diff"], 5e-4)
        self.assertLessEqual(bench["J1_max_abs_diff"], 5e-2)
        self.assertLessEqual(bench["J0_max_abs_diff"], 5e-2)


if __name__ == "__main__":
    unittest.main()
