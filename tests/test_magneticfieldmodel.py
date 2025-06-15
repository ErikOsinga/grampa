from __future__ import annotations

import unittest

import numpy as np

from grampa import magneticfieldmodel
from grampa import magneticfieldmodel_utils as mutils

rng = np.random.default_rng(42)  # reproducibility


def assign_default():
    return {
        "sourcename": "test",
        "reffreq": 944,
        "cz": 0.021,
        "xi": 5.67,
        "N": 256,
        "pixsize": 10.0,
        "eta": 0.5,
        "B0": 1.0,
        "lambdamax": None,
        "dtype": 32,
        "garbagecollect": True,
        "iteration": 0,
        "beamsize": 20.0,
        "recompute": True,
        "ne0": 0.0031,
        "rc": 341,
        "beta": 0.77,
        "fluctuate_ne": True,
        "mu_ne_fluct": 1.0,
        "sigma_ne_fluct": 0.2,
        "lambdamax_ne_fluct": 100,
        "subcube_ne": True,
        "testing": False,
        "savedir": "./testdata",
        "saverawfields": True,
        "saveresults": True,
        "redshift_dilution": True,
        "nthreads_fft": 6,
        "frame": "observedframe",
    }


def ne_funct(r):
    """Takes radius in kpc and returns n_e in cm^-3 (Osinga+24 parameters for Abell 2256)."""
    return mutils.beta_model(r, ne0_cm3=0.0031, rc_kpc=341, beta=0.77)


def approximate_slope_in_loglog_space(k_vals, P_vals):
    """
    Returns the slope of P(k) in log-log space.
    That is, fits log(P) = slope * log(k) + intercept.
    """
    log_k = np.log(k_vals)
    log_P = np.log(P_vals)
    slope, intercept = np.polyfit(log_k, log_P, 1)
    return slope, intercept


class TestMagneticFieldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Class-level setup called once for all test methods.
        """
        cls.args = assign_default()
        cls.model = magneticfieldmodel.MagneticFieldModel(cls.args, ne_funct=ne_funct)
        cls.model.run_model()

    def test_Bfield_generation(self):
        """
        Test if magnetic field generation produces expected output:
            scaling as B0*density^eta
            correct power spectrum k^-n.
        """

        # Check NaN and reasonable values
        self.assertFalse(
            np.isnan(self.model.B_field_norm).any(),
            "B-field array should not contain NaNs.",
        )
        self.assertFalse(
            np.isnan(self.model.RMimage).any(), "RM image should not contain NaNs."
        )
        self.assertTrue(
            np.all(self.model.Polint <= 1) & np.all(self.model.Polint >= 0),
            "Polarization intensity should be between 0 and 1.",
        )
        self.assertFalse(
            np.any(np.abs(self.model.RMimage) > 1e5),
            "Faraday rotation measure should not exceed 1e5 for a small cube.",
        )

        # Check if B-field follows electron density profile reasonably well (there will be fluctuations)
        all_r, profile, density = mutils.plot_Bfield_amp_vs_radius(
            self.model.B_field_norm,
            self.model.pixsize,
            self.model.ne_funct,
            self.model.B0,
            savefig="./test_Bfield_vs_ne.png",
            show=False,
        )
        self.assertIsNotNone(all_r, "Radius array should not be None.")
        self.assertIsNotNone(profile, "Profile array should not be None.")
        self.assertEqual(
            len(all_r),
            len(profile),
            "Radius and profile arrays must be of equal length.",
        )
        self.assertGreater(len(all_r), 0, "Radius array should not be empty.")
        self.assertTrue(
            np.all(profile >= 0), "Magnetic field amplitude should not be negative."
        )
        self.assertTrue(np.all(density >= 0), "Density values should not be negative.")

        expect_B = ((density / density[0]) ** 0.5) * self.model.B0
        self.assertTrue(
            np.allclose(np.mean(expect_B / profile), 1, rtol=2),
            "B-field does not follow density profile.",
        )

        # Check if B-field follows power spectrum reasonably well
        if self.model.lambdamax < self.model.N * self.model.pixsize / 2:
            raise NotImplementedError(
                "Did not implement test for power spectrum for lambdamax not equal to maximum scale"
            )
        k_values, Pk_values, theoretical = mutils.plot_B_field_powerspectrum(
            self.model.B_field_norm,
            self.model.xi,
            self.model.lambdamax,
            savefig="./test_Bfield_powerspectrum.png",
            show=False,
        )
        slope, _ = approximate_slope_in_loglog_space(k_values, Pk_values)
        # note that power spectrum n = xi-2 is defined with negative exponent: Bk ~ k^-n
        slope *= -1
        self.assertTrue(
            np.isclose(slope, self.model.xi - 2, rtol=0.1),
            "Power spectrum slope does not match expected value to within 10%.",
        )

    # Additional tests will reuse the initialized model without re-running setup

    def test_other_things(self):
        print("MORE TESTS TODO")


if __name__ == "__main__":
    unittest.main()

    """
    test = TestMagneticFieldModel()
    test.setUpClass()
    test.test_Bfield_generation()
    """
