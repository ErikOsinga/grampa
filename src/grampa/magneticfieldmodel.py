from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from collections.abc import Callable

import astropy.units as u
import numpy as np
import psutil
import pyfftw
from astropy.constants import c as speed_of_light
from astropy.cosmology import FlatLambdaCDM

from grampa import magneticfieldmodel_utils as mutils

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # TODO, make optional

"""
# Simulate a magnetic field following Murgia+2004; https://arxiv.org/abs/astro-ph/0406225
# approach is first mentioned in Tribble+1991.
# Assumes spherical symmetry in the electron density and magnetic field profile.

# Code was developed for Osinga+22 and Osinga+25, and the lognormal electron density field was added by Khadir+25

### TODO
# 1. worth looking into DASK for chunking large array multiplications in memory
#    (e.g. https://docs.dask.org/en/stable/generated/dask.array.dot.html)
# 2. Better logging (INFO, DEBUG, ERROR, WARNING)
# 3. Lambda_max=None case should be an integer instead of None

"""

__version__ = "0.0.1"


class MagneticFieldModel:
    def __init__(self, args: dict, ne_funct: Callable) -> None:
        """
        Initialise the MagneticFieldModel

        args     -- dict     -- Contains the parameters, see the end of this file for definitions.
        ne_funct -- Callable -- Should be a callable (function) taking radius [kpc] as the argument and returning electron density [cm^-3]

        returns None
        """
        self.starttime = time.time()
        pid = os.getpid()
        self.python_process = psutil.Process(pid)

        self.logger = logging.getLogger(__name__)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.logfile = f"grampa_logs_{timestr}.log"

        # File handler
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
            )
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
            )
        )

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Set level
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating magnetic field model with GRAMPA...")

        self.args = args
        self.sourcename = args["sourcename"]
        self.reffreq = args["reffreq"]
        self.cz = args["cz"]
        self.N = args["N"]
        self.pixsize = args["pixsize"]
        self.xi = args["xi"]
        self.eta = args["eta"]
        self.B0 = args["B0"]
        self.lambdamax = args["lambdamax"]  # user input lambdamax, can be None
        self.lambdamax = self.check_lambdamax(
            self.lambdamax
        )  # convert lambdamax to a number in kpc if None
        self.dtype = args["dtype"]
        self.check_ftype_ctype()
        self.garbagecollect = args["garbagecollect"]
        self.iteration = args["iteration"]
        self.beamsize = args["beamsize"]
        self.recompute = args["recompute"]
        if "ne0" in args.keys():
            self.ne0 = args["ne0"]
        if "rc" in args.keys():
            self.rc = args["rc"]
        if "beta" in args.keys():
            self.beta = args["beta"]
        self.fluctuate_ne = args["fluctuate_ne"]
        self.mu_ne_fluct = args["mu_ne_fluct"]
        self.sigma_ne_fluct = args["sigma_ne_fluct"]
        self.lambdamax_ne_fluct = args["lambdamax_ne_fluct"]
        self.lambdamax_ne_fluct = self.check_lambdamax(
            self.lambdamax_ne_fluct
        )  # convert to a number in kpc if None
        self.testing = args["testing"]
        self.savedir = args["savedir"]
        self.saverawfields = args["saverawfields"]
        self.saveresults = args["saveresults"]
        self.frame = args["frame"]
        self.subcube_ne = args["subcube_ne"]
        self.nthreads_fft = 48
        self.ne_funct = ne_funct

        # Check resolution and beam size
        self.check_resolution_beamsize()

        # Create paramstring for saving files
        self.paramstring = self.create_paramstring()

        # Check directory where outputs will be saved
        self.check_savedir()

        # log parameters to file
        self.log_parameters()

    def check_lambdamax(self, lambdamax: float | None) -> float:
        if lambdamax is None:
            # Smallest possible k mode (k=1) corresponds to Lambda=(N*pixsize)/2, one reversal
            lambdamax = (self.N * self.pixsize) / 2
        elif lambdamax > (self.N * self.pixsize) / 2:
            lambdamax = (self.N * self.pixsize) / 2
            self.logger.warning(
                f"Warning: Input Lambda_max is larger than the maximum possible scale. Setting Lambda_max to maximum possible scale of {self.lambdamax} kpc."
            )
        elif lambdamax < 0:
            errormsg = f"Error: {lambdamax=} but it cannot be negative."
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        return lambdamax

    def check_savedir(self):
        """Check if savedir exists, if not create it"""
        if self.savedir[-1] != "/":
            self.savedir += "/"
        if not os.path.exists(self.savedir):
            self.logger.info(f"Creating output directory {self.savedir}")
            os.mkdir(self.savedir)
        return

    def check_ftype_ctype(self):
        if self.dtype == 32:
            self.ftype = np.float32
            self.ctype = np.complex64
        elif self.dtype == 64:
            self.ftype = np.float64
            self.ctype = np.complex128
        else:
            errormsg = (
                f"Cannot set dtype to float{self.dtype}. Valid values are 32 or 64."
            )
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        return

    def check_resolution_beamsize(self):
        # Calculate the resolution we have at cluster redshift with an X arcsec beam
        resolution = (
            cosmo.kpc_proper_per_arcmin(self.cz) * self.beamsize * u.arcsec
        ).to(u.kpc)
        self.FWHM = resolution.value  # in kpc

        if (
            self.FWHM < self.pixsize * 5
        ):  # 5 is approximately 2 * 2sqrt(2ln(2))  (because we want at least 2 pix)
            # Set it automatically so the FWHM corresponds to 5 * pixsize at cluster redshift
            self.logger.warning(
                f"User input angular resolution of {self.beamsize} arcsec corresponds to physical resolution of {self.FWHM:.2f} kpc (FWHM)."
            )
            self.FWHM = self.pixsize * 5  # kpc
            self.beamsize = (
                (
                    self.FWHM
                    * u.kpc
                    / (cosmo.kpc_proper_per_arcmin(self.cz).to(u.kpc / u.arcsec))
                )
                .to(u.arcsec)
                .value
            )
            self.logger.warning(
                f"WARNING: However, models are being ran with p={self.pixsize} kpc. The code will smooth to {self.FWHM} kpc automatically. This corresponds to a beam size of {self.beamsize:.2f} arcsec instead. Please keep this in mind."
            )
        return

    def create_paramstring(self):
        if self.lambdamax is None:
            paramstring = f"N={self.N}_p={self.pixsize}_B0={self.B0:.1f}_xi={self.xi:.2f}_eta={self.eta:.2f}_s={self.sourcename}_it{self.iteration}_b{self.beamsize:.2f}asec"
        else:
            paramstring = f"N={self.N}_p={self.pixsize}_B0={self.B0:.1f}_xi={self.xi:.2f}_eta={self.eta:.2f}_s={self.sourcename}_Lmax={self.lambdamax}_it{self.iteration}_b{self.beamsize:.2f}asec"
        if self.frame == "observedframe":
            paramstring += "_obsframe"
        elif self.frame == "clusterframe":
            paramstring += "_clusterframe"
        else:
            errormsg = f"Error: {self.frame=} but it can only be 'observedframe' or 'clusterframe'."
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        if self.fluctuate_ne:
            paramstring += f"_neLmax_{self.lambdamax_ne_fluct:.0f}"

        return paramstring

    def log_parameters(self):
        self.logger.info("Using parameters:")
        self.logger.info(f" xi={self.xi:.2f} (n={self.xi - 2:.2f})")
        self.logger.info(f" N={self.N}")
        self.logger.info(f" eta={self.eta:.1f}")
        self.logger.info(f" B0={self.B0:.1f}")
        self.logger.info(f" pixsize={self.pixsize:.1f} kpc")
        self.logger.info(f" sourcename= {self.sourcename}")
        self.logger.info(f" cz= {self.cz:.2f}")
        self.logger.info(f" Lambda_max= {self.lambdamax} kpc")
        self.logger.info(f" Beam FWHM = {self.beamsize:.1f} arcsec")
        self.logger.info(f" Beam FWHM = {self.FWHM:.1f} kpc")
        self.logger.info(f" dtype= float{self.dtype}")
        self.logger.info(f" Manual garbagecollect= {self.garbagecollect}")
        if "ne0" in self.args.keys():
            self.logger.info(f" ne0= {self.ne0:.2f} cm^-3")
        if "rc" in self.args.keys():
            self.logger.info(f" rc= {self.rc:.2f} kpc")
        if "beta" in self.args.keys():
            self.logger.info(f" beta= {self.beta:.2f}")
        self.logger.info(f" testing= {self.testing}")
        self.logger.info(f" Fluctuate ne = {self.fluctuate_ne}")
        if self.fluctuate_ne:
            if self.mu_ne_fluct != 1:
                self.logger.info(f" mu_ne_fluct = {self.mu_ne_fluct:.2f}")
            self.logger.info(f" sigma_ne_fluct = {self.sigma_ne_fluct:.2f}")
            self.logger.info(f" ne fluct Lmax = {self.lambdamax_ne_fluct:.0f} kpc")

        self.logger.info(f" savedir= {self.savedir}")
        self.logger.info(f" paramstring= {self.paramstring}")

    def check_results_already_computed(self):
        """
        Check whether we already have a 2D RM image with the current parameters
        or perhaps with a different value of B_0

        RETURNS
        a string that is either
            'fully computed'     -- RM images are already computed with the given B0
            'partially computed' -- RM images are already computed with B0=1, but not with the given B0
            'not computed'       -- RM images are not yet computed (might have pre-normalised B field)
        """
        savedir2 = f"{self.savedir}after_normalise/{self.sourcename}/"

        # First check if the result with the given B0 is already computed
        if os.path.isfile(
            f"{savedir2}RMimage_{self.paramstring}.npy"
        ) and os.path.isfile(f"{savedir2}RMhalfconvolved_{self.paramstring}.npy"):
            return "fully computed"
        else:
            # Check if the result with B0=1 is already computed. We can use it
            # to compute the result with any other B0
            if os.path.isfile(
                f"{savedir2}RMimage_{self.paramstring}.npy"
            ) and os.path.isfile(f"{savedir2}RMhalfconvolved_{self.paramstring}.npy"):
                return "partially computed"
            else:
                return "not computed"

    def run_model(self):
        """
        Compute a model RM and depolarisation image for a given set of magnetic field parameters.

        Will create the following class variables:

        self.B_field_norm      -- array of shape (N,N,N,3) -- normalised B field cube, all three vector dimensions
        self.RMimage           -- array of shape (N,N)     -- RM for a source behind the cluster (full path length) (in cluster or observed frame)
        self.RMimage_half      -- array of shape (N,N)     -- RM for a source half-way inside the cluster (half path length) (in cluster or observed frame)
        self.RMconvolved       -- array of shape (N,N)     -- self.RMimage convolved to the user requested resolution
        self.RMhalfconvolved   -- array of shape (N,N)     -- self.RMimage_half convolved to the user requested resolution
        self.Qconvolved        -- array of shape (N,N)     -- Stokes Q convolved to the user requested resolution
        self.Uconvolved        -- array of shape (N,N)     -- Stokes U convolved to the user requested resolution
        self.Qconvolved_half   -- array of shape (N,N)     -- As above, but with integration through half the cluster
        self.Uconvolved_half   -- array of shape (N,N)     -- As above, but with integration through half the cluster
        self.Polint            -- array of shape (N,N)     -- Polarisation intensity fraction after convolution (wrt 1[unit] uniform angle intrinsically polarised radiation)
        self.Polint_inside     -- array of shape (N,N)     -- As above, but with integration through half the cluster
        self.coldens           -- array of shape (N,N)     -- Electron column density image
        self.Bfield_integrated -- array of shape (N,N)     -- Magnetic field integrated along the line of sight
        """

        # First check whether the results are already computed
        status = self.check_results_already_computed()

        if not self.recompute and status == "fully computed":
            dtime = time.time() - self.starttime
            self.logger.info(
                f"Script fully finished. Took {dtime:.1f} seconds to check results"
            )
            self.logger.info("Results already computed and recompute=False, exiting.")
            sys.exit("Results already computed and recompute=False, exiting.")

        # Otherwise status = partially computed or not computed, continue.
        if not self.recompute and status == "partially computed":
            self.logger.info(
                f"Loading RM image from file with B0=1, and scaling it to B0={self.B0}"
            )  # todo, load any field strength
            RMimage, RMimage_half, RMconvolved, RMhalfconvolved = (
                self.computeRMimage_from_file()
            )

        else:  # otherwise we need to compute the RM images, could be from scratch or from vectorpotential or Bfield
            # Check whether the vector potential file or Bfield file already exists, would save time
            (
                already_computed_Afield,
                already_computed_Bfield,
                vectorpotential_file,
                Bfield_file,
            ) = self.check_results_computed()

            if self.recompute:
                self.logger.info("User forces recompute, so recomputing everything.")
                already_computed_Afield = False
                already_computed_Bfield = False

            if already_computed_Bfield:
                self.logger.info(
                    "Found a saved version of the (pre-normalisation) magnetic field with user defined parameters."
                )
                self.logger.info(
                    f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}"
                )
                self.logger.info("Loading from file..")
                B_field = np.load(Bfield_file)

            elif already_computed_Afield:
                self.logger.info(
                    "Found a saved version of the vector potential with user defined parameters."
                )
                self.logger.info(
                    f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}"
                )
                self.logger.info("Loading from file..")

                # then compute B from A
                B_field = self.computeBfromA(vectorpotential_file)

            else:
                if not self.recompute:
                    self.logger.info(
                        "No saved version of the magnetic field or vector potential found."
                    )
                self.logger.info("Starting from scratch...")

                # compute the vector potential
                field = mutils.calculate_vectorpotential(
                    self.N, self.xi, self.lambdamax, self.pixsize, self.ftype
                )

                # compute B from A
                B_field = self.computeBfromA(None, field)

            # If they were not already computed, we'll save them to file for future use
            if not already_computed_Afield and self.saverawfields:
                self.logger.info(
                    f"Saving fourier vector potential to {vectorpotential_file}, such that it can be used again"
                )
                np.save(vectorpotential_file, field)
            if not already_computed_Bfield and self.saverawfields:
                self.logger.info(
                    f"Saving unnormalised magnetic field to {Bfield_file}, such that it can be used again"
                )
                np.save(Bfield_file, B_field)

                self.logger.debug(f"Resulting magnetic field shape: {B_field.shape}")

            ########## Normalisation of the B field with some density profile ##########
            self.logger.info("Normalising profile with electron density profile")

            ## Using radial symmetry in a way where we can only use 1/8th of the cube
            ## we can calculate ne_3d about 6x faster for N=1024
            self.logger.info(
                f"Using subcube symmetry to speed up calculations: {self.subcube_ne}"
            )

            # Vector denoting the real space positions. The 0 point is in the middle.
            # e.g. runs from -31 to +32 (for N=64). Or 0 to +32 when subcube=True and we use symmetry
            # Then only take the norm of the position vector
            xvec_length = mutils.xvector_length(
                self.N, 3, self.pixsize, self.ftype, subcube=self.subcube_ne
            )

            if self.fluctuate_ne:
                self.logger.info("Generating ne cube with fluctuations")
                # Generate lognormal fluctuations in ne with some standard deviation
                ne_3d = mutils.gen_ne_fluct(
                    self.xi,
                    self.N,
                    self.pixsize,
                    self.mu_ne_fluct,
                    self.sigma_ne_fluct,
                    Lambda_max=self.lambdamax_ne_fluct,
                    Lambda_min=None,
                    indices=True,
                )

                self.logger.info(
                    "Normalising ne cube to follow the mean profile of the requested beta function"
                )
                ne_3d = mutils.normalise_ne_field(
                    xvec_length, ne_3d, self.ne_funct, subcube=self.subcube_ne
                )
                # ne_3d is now always of shape (N,N,N), regardless of whether subcube=True because of the spatial variations

            else:
                # Generate the electron density field without fluctuations
                ne_3d = self.ne_funct(xvec_length)
                # ne_3d can now be (N//2+1,N//2+1,N//2+1), if subcube=True, because spherically symmetric

            del xvec_length  # We dont need xvec_length anymore

            if self.subcube_ne and not self.fluctuate_ne:
                c = 0  # then the center pixel of ne_3d is the first pixel, because the subcube is only the positive subset
                subcube_for_B = (
                    True  # can calculate B normalisation with the efficient function
                )
            else:
                c = self.N // 2 - 1  # then the center pixel is at N//2-1
                subcube_for_B = False  # cannot calculate B normalisation with the efficient function

            # Make sure n_e is not infinite in the center (in case diverging ne model at r=0). Just set it to the pixel next to it
            ne_3d[c, c, c] = ne_3d[c, c + 1, c]
            ne0 = ne_3d[c, c, c]  # Electron density in center of cluster

            # Normalise the B field such that it follows the electron density profile ^eta
            B_field_norm, ne_3d = mutils.normalise_Bfield(
                ne_3d, ne0, B_field, self.eta, self.B0, subcube_for_B
            )

            del B_field  # We dont need B field unnormalised anymore, its quite big
            if self.garbagecollect:
                self.logger.info(
                    "Deleted B_field and xvec_length. Collecting garbage.."
                )
                gc.collect()
                memoryUse = self.python_process.memory_info()[0] / 2.0**30
                self.logger.info(f"Memory used: {memoryUse:.1f} GB")

            # Save the B_field_norm as a class variable
            self.B_field_norm = B_field_norm

            # Calculate the B_field amplitude (length of the vector)
            # B_field_amplitude_nonorm = np.copy(B_field_amplitude)
            # B_field_amplitude = np.linalg.norm(B_field_norm,axis=3)

            if self.testing:
                print("Plotting normalised B-field amplitude")
                mutils.plot_Bfield_amp_vs_radius(
                    B_field_norm, self.pixsize, self.ne_funct, self.B0
                )
                print("Plotting normalised B-field power spectrum")
                mutils.plot_B_field_powerspectrum(B_field_norm, self.xi, self.lambdamax)

            self.logger.info("Calculating rotation measure images.")
            # now we make full 3D density cube anyway to calculate rotation measure image # TODO: could improve efficiency also here
            if subcube_for_B:
                ne_3d = mutils.cube_from_subcube(ne_3d, self.N, self.ftype)

            if self.testing:
                print(f"Plotting electron density image slice, shape {ne_3d.shape}")
                mutils.plot_ne_image(
                    ne_3d[:, :, self.N // 2 - 1], self.pixsize, title="Slice of ne_3d"
                )
                mutils.plot_ne_image(
                    np.log10(ne_3d[:, :, self.N // 2 - 1]),
                    self.pixsize,
                    title="Slice of log10(ne_3d)",
                )
                mutils.plot_ne_profile(
                    ne_3d[:, :, self.N // 2 - 1],
                    self.pixsize,
                    self.ne_funct,
                    title="ne profile vs user function",
                )

            # Calculate the RM by integrating over the 3rd axis
            RMimage = mutils.RM_integration(ne_3d, B_field_norm, self.pixsize, axis=2)
            # Also integrate over half of the third axis. For in-cluster sources
            RMimage_half = mutils.RM_halfway(ne_3d, B_field_norm, self.pixsize, axis=2)

            # Correct for redshift dilution if RM is requested in observed frame
            if self.frame == "observedframe":
                RMimage = RMimage / (1 + self.cz) ** 2
                RMimage_half = RMimage_half / (1 + self.cz) ** 2

            # Convolve the RM image with the requested resolution.
            # From here we can start to use float64 again, because the images are 2D
            RMconvolved, RMhalfconvolved = mutils.convolve_with_beam(
                [RMimage, RMimage_half], self.FWHM, self.pixsize
            )

            # Save the RM images as class variables
            self.RMimage = RMimage
            self.RMimage_half = RMimage_half
            self.RMconvolved = RMconvolved
            self.RMhalfconvolved = RMhalfconvolved

        # Now we have the RM images either from rescaling or from scratch
        if self.testing:
            print("Plotting RM images. Unconvolved & convolved")
            mutils.plotRMimage(RMimage, self.pixsize, title="RM image not convolved")
            mutils.plotRMimage(RMconvolved, self.pixsize, title="RM image convolved")
            print("Plotting RM power spectrum")
            mutils.plot_RM_powerspectrum(
                RMimage, self.xi, self.lambdamax, title="RM image not convolved"
            )
            mutils.plot_RM_powerspectrum(
                RMconvolved, self.xi, self.lambdamax, title="RM image convolved"
            )

        # Calculate Stokes Q and U images at the centre frequency
        # rotate their polarisation angle with the RM map, and convolve them
        # to produce beam depolarisation

        # Randomly set an intrinsic polarisation angle (uniform)
        phi_intrinsic = 45 * np.pi / 180  # degrees to radians
        # Observed wavelength of radiation in meters
        wavelength = (speed_of_light / (self.reffreq * u.MHz)).to(u.m).value

        # Calculate observed polarisation angle by rotating the intrinsic angle
        if (
            self.frame == "observedframe"
        ):  # phi_obs = phi_intrinsic + RM_obs*lambda_obs^2
            phi_obs = mutils.calc_phi_obs(
                phi_intrinsic, RMimage, wavelength
            )  # Shape (N,N)
            phi_obs_inside = mutils.calc_phi_obs(
                phi_intrinsic, RMimage_half, wavelength
            )
        elif (
            self.frame == "clusterframe"
        ):  # phi_obs = phi_intrinsic + RM_cluster*lambda_cluster^2
            wavelength_cluster = wavelength / (1 + self.cz)
            phi_obs = mutils.calc_phi_obs(phi_intrinsic, RMimage, wavelength_cluster)
            phi_obs_inside = mutils.calc_phi_obs(
                phi_intrinsic, RMimage_half, wavelength_cluster
            )

        # Convert the pol angle and polarised intensity (constant) to Stokes Q and U
        polint_intrinsic = 1  # Arbitrary value, now polint will correspond to polarisation fraction p/p0
        self.logger.info("Calculating Stokes Q and U images")
        Qflux, Uflux = mutils.StokesQU_image(phi_obs, polint_intrinsic)
        # Also for a screen inside the cluster (less rotation)
        Q_half, U_half = mutils.StokesQU_image(phi_obs_inside, polint_intrinsic)

        self.logger.info("Convolving Stokes Q and U images")
        self.Qconvolved, self.Uconvolved = mutils.convolve_with_beam(
            [Qflux, Uflux], self.FWHM, self.pixsize
        )
        self.Qconvolved_half, self.Uconvolved_half = mutils.convolve_with_beam(
            [Q_half, U_half], self.FWHM, self.pixsize
        )

        # polangle = np.arctan2(Uconvolved,Qconvolved)*0.5
        self.Polint = np.sqrt(self.Qconvolved**2 + self.Uconvolved**2)
        # polangle_inside = np.arctan2(Uconvolved_half,Qconvolved_half)*0.5
        self.Polint_inside = np.sqrt(self.Qconvolved_half**2 + self.Uconvolved_half**2)

        # Calculate the column density image
        self.coldens = mutils.columndensity(ne_3d, self.pixsize, axis=2)

        # and integrate B field along the LOS
        self.Bfield_integrated = np.sum(B_field_norm[:, :, :, 2], axis=2)

        dtime = time.time() - self.starttime
        self.logger.info(
            f"Script calculations finished. Took {dtime:.1f} seconds which is {dtime / 3600.0:.1f} hours or {dtime / 86400.0:.1f} days"
        )

        if self.testing:
            # Plot depol images
            mutils.plotdepolimage(self.Polint, self.pixsize, "Polint full pathlength")
            mutils.plotdepolimage(
                self.Polint_inside, self.pixsize, "Polint half pathlength"
            )

        if self.saveresults:
            self.save_results()
        else:
            self.logger.info("Not saving results to file.")

        dtime = time.time() - self.starttime
        self.logger.info(
            f"Script fully finished. Took {dtime:.1f} seconds which is {dtime / 3600.0:.1f} hours or {dtime / 86400.0:.1f} days"
        )

        return

    def computeRMimage_from_file(self):
        """
        If we already have an RM image with B0=1, we can simply scale it to any other B0
        because we're simply doing  X * integral(B*ne) dr = X * RM
        """
        savedir2 = self.savedir + f"after_normalise/{self.sourcename}/"
        # Load the B0=1 results # TODO, can be any B0

        RMimage = np.load(f"{savedir2}RMimage_{self.paramstring}.npy")
        RMimage_half = np.load(f"{savedir2}RMimage_half_{self.paramstring}.npy")
        RMconvolved = np.load(f"{savedir2}RMconvolved_{self.paramstring}.npy")
        RMhalfconvolved = np.load(f"{savedir2}RMhalfconvolved_{self.paramstring}.npy")

        # Scale with whatever B0 we have now
        RMimage *= self.B0
        RMimage_half *= self.B0
        RMconvolved *= self.B0
        RMhalfconvolved *= self.B0

        return RMimage, RMimage_half, RMconvolved, RMhalfconvolved

    def computeBfromA(
        self, vectorpotential_file: str, field: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute the magnetic field from the vector potential. Cross product of k and A
        """
        if field is None:
            # Load vector potential file
            field = np.load(vectorpotential_file)

        self.logger.info(
            f"Generating k vector in ({self.N},{self.N},{self.N // 2},3) space"
        )
        kvec = mutils.kvector(
            self.N, 3, self.pixsize, self.ftype
        )  # 3 = number of dimensions of the vector field

        # Fourier B field = Cross product  B = ik \cross A
        self.logger.info("Calculating magnetic field using the crossproduct Equation")
        field = mutils.magnetic_field_crossproduct(kvec, field, self.N, self.ctype)
        del kvec  # Huge array which we dont need anymore
        if self.garbagecollect:
            self.logger.info("Deleted kvec. Collecting garbage..")
            gc.collect()
            memoryUse = self.python_process.memory_info()[0] / 2.0**30
            self.logger.info(f"Memory used: {memoryUse:.1f} GB")

        # B field is the inverse fourier transform of fourier_B_field
        run_ift = pyfftw.builders.irfftn(
            field,
            s=(self.N, self.N, self.N),
            axes=(0, 1, 2),
            auto_contiguous=False,
            auto_align_input=False,
            avoid_copy=True,
            threads=self.nthreads_fft,
        )
        field = run_ift()
        # memoryUse = self.python_process.memory_info()[0]/2.**30
        # self.logger.info('Memory used: %.1f GB'%memoryUse)
        if self.garbagecollect:
            self.logger.info("Ran IFFT.. Collecting garbage..")
            gc.collect()
            memoryUse = self.python_process.memory_info()[0] / 2.0**30
            self.logger.info(f"Memory used: {memoryUse:.1f} GB")

        return field

    def check_results_computed(self):
        """ """
        # The files where the vector potential and B field are / will be saved
        vectorpotential_file, Bfield_file = self.BandAfieldfiles()

        # Boolean to track whether maybe we have computed unnormalised A or B field before
        already_computed_Afield = False
        already_computed_Bfield = False
        if os.path.isfile(vectorpotential_file) and not self.recompute:
            already_computed_Afield = True
            self.logger.info(
                "Found a saved version of the vector potential with user defined parameters:"
            )
            self.logger.info(
                f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}"
            )

            self.logger.info("Checking if magnetic field was also already computed..")
            if os.path.isfile(Bfield_file):
                self.logger.info("Magnetic field already computed in a previous run")
                already_computed_Bfield = True
            else:
                self.logger.info("Magnetic field not computed in a previous run")

        return (
            already_computed_Afield,
            already_computed_Bfield,
            vectorpotential_file,
            Bfield_file,
        )

    def BandAfieldfiles(self):
        # TODO: if lambda_max is the maximum scale of the grid
        # then the power spectrum is scale invariant, so pixel size is not important
        # could save the B and A field files without pixsize in the name

        vectorpotential_file = f"{self.savedir}Afield_N={self.N:.0f}_p={self.pixsize:.0f}_xi={self.xi:.2f}_Lmax={self.lambdamax:.0f}_it{self.iteration}.npy"
        Bfield_file = f"{self.savedir}Bfield_N={self.N:.0f}_p={self.pixsize:.0f}_xi={self.xi:.2f}_Lmax={self.lambdamax:.0f}_it{self.iteration}.npy"

        return vectorpotential_file, Bfield_file

    def save_results(self):
        savedir2 = self.savedir + f"after_normalise/{self.sourcename}/"
        if not os.path.exists(savedir2):
            os.makedirs(savedir2)

        self.logger.info(f"Saving results to {savedir2}")

        # Save the RM images
        np.save(f"{savedir2}RMimage_{self.paramstring}.npy", self.RMimage)
        np.save(f"{savedir2}RMimage_half_{self.paramstring}.npy", self.RMimage_half)
        np.save(f"{savedir2}RMconvolved_{self.paramstring}.npy", self.RMconvolved)
        np.save(
            f"{savedir2}RMhalfconvolved_{self.paramstring}.npy", self.RMhalfconvolved
        )

        # Save the Stokes Q and U images
        np.save(f"{savedir2}Qconvolved_{self.paramstring}.npy", self.Qconvolved)
        np.save(f"{savedir2}Uconvolved_{self.paramstring}.npy", self.Uconvolved)

        # Save the column density and B field images
        np.save(f"{savedir2}coldens_{self.paramstring}.npy", self.coldens)
        np.save(
            f"{savedir2}Bfield_integrated_{self.paramstring}.npy",
            self.Bfield_integrated,
        )

        # Dont save the normalised B field, it's a 3Dx3 cube so its large
        # np.save(f"{savedir2}Bfield_{self.paramstring}.npy", self.B_field_norm)

        # Save the polarisation intensity
        np.save(f"{savedir2}Polint_{self.paramstring}.npy", self.Polint)
        np.save(f"{savedir2}Polint_inside_{self.paramstring}.npy", self.Polint_inside)

        return


def str2bool(v):
    """
    Parse input of the form

    -fluctuate_ne True
    -fluctuate_ne False

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean type expected.")


def welcome_ascii() -> str:
    return r"""
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ......................:=+++++++=-..................................-=+++++++=:......................
    ..................:=++=-:.....:-=+++-..........................-+++=-:.....:-=++=:..................
    ................:++-...............:=+=:....................:=+=:...............-++-................
    ...............=+-....................=+=.................:=+=....................:+=...............
    .............:+=:......:=+++===+++=:....=+:..............:+=....:=+++===+++=:......:=+:.............
    ............:++......:=+-:........-++=...-+-...::--::...-+-...-+=-........:-++:......++:............
    ............=+......=+-.............:=+-..=##%%#****#%###=..-+=:.............-+=......+=:...........
    ...........-+:.....=+:......:-=++=-:..:*###=-::::::::::-+#%#*:...:==+=-:.......+=.....:+-...........
    ...........+=.....=+:.....-++-::::-++=*%+-::::::::::::::::-+%*-++-::::-=+-.....:+=.....-+...........
    ..........-+-....-+-.....++:........=%+-::::::::::::::::::::-*%=........:++:....-+-....:+-..........
    ..........++.....++....:++.......::=%+-::::::::::::::::::::::-+%-::.......=+:....=+:....=+..........
    .........:+=....-+:....=+.....-++=+%*-::::::::::::::::::::::::-*%===+=.....+=....:+-....-+:.........
    .........-+-....=+....:+:....+=:..+%-::::::::::::::::::::::::::-%+...=+....:+:....+=....:+-.........
    .........-+:...:==....==....=+:..:**::::::::::::::::::::::::::::#*:..:==....==....=+:...:+-.........
    .........=+....:+-....+-...-+-...-#+:::-+***+-::-:::::=****+-:::*#:...-+:...-+....:+:....+=.........
    .........=+....:+:...:+-...=+....-#=::-#*+==+=::-::::-++==+##:::+#-...:+=...:+:...:+:....+=.........
    .........=+....:+....-+:...=+..=*#%=---=#####+---::::-*####*---:+%#*-..+=...:+-....+-....+=.........
    .........=+....-*:...-+:...++.=#-=#%#*#*-:::-+%==++=+%=--:-=#**#%#==%=.*+...:+-....*-....++.........
    .........+*....-*:...-*:..:++:+*::==-+#-:+##=-##=--+#+-=##=-=#=-=-:-#+.*+...:*-....#-....*+.........
    .........+*....-#:...-*-...+*.+#:::::=#=:-++--%#----**--=+--+#-:----%=.#+...-*-....#-....*+.........
    .........=#....-*:...:*-...+#:-#=::::-=%*===+#%+-----##+===*#-:::-:+#::#=...-*:...:*-....#+.........
    .........=#:...:*-....#=...-*-.*#-::::---=++=+%--------=++=---::--=#=.=#-...=*....-*-...:#=.........
    .........-#-...:+=....++....*+..+%%*::-:--=*##%+---=#%##+-------#%%=.:+*....++....=*:...-#=.........
    .........:*=....+*....-*:...:*=...*%-:--*#+:...=*#*+:...-*#+---=%+...=*:...:*-....*+....=#-.........
    ..........*+....-*:...:++....-*+..+#+##*-..................=###+%-..+*-....*+:...-#=....+#..........
    ..........+*:...:*+....-*=....:+#++%*:..........-#+:..........-#%=+**:....=*-....+*:...:*+..........
    ..........:*=....=*-....-*=......--+%%*+=---=+*%#++##*=----=+#%%=--......=*-....-*=....-*:..........
    ...........++:....++:....:**:.......*#=-=+++==--::----=+++==-=#*.......:+*-....:++....:++...........
    ...........:*-.....*+......-***==+****#-::::::+#%%%##=::--:-=%*+**+==***=......+*:....-*:...........
    ............=*:....:+*:.......:--:...-#%+-:::::::-::::-::--*%*:...:--:.......-*+:....:*=............
    .............+*:.....=*+:..........:+*-:*#+-:::::::::-:--*#+:=*+:..........:+*=.....:*+.............
    ..............**-......=*#+-:::-=*#*-..:***%#+--------*#%+**...-*#*=-:::=***=......:**..............
    ...............=*=........:-====-:...:+*-...:-+**##**+-....=#+:...:-====-:........=*=...............
    ................:+*=...............-**-......................=**-..............:=**:................
    ...................=*#=:.......:=***:..........................-***=:.......:+#*+...................
    ......................-++*****++-:................................:-++*****+=-......................
    ....................................................................................................
    ....................................................................................................
    ................:*%%%%%*:..*%%%%%#+.....:#%%+.....#%%=....-#%#-..*%%%%%%+:...=%%%-..................
    ...............-#%*:.:#%#..*%#::-#%*....+%#%%-....#%%%-...*%%#=..*%#::-%%+..:#%#%#..................
    ...............+%%.........*%#:..*%#...:%%:+%#....#%#%#:.+%#%%=..*%*...%%*:.+%%:*%+.................
    ...............+%%..+%%%#..*%%%%%%*:...#%*.:#%=...%%=*%+=%#=%%=..*%%#%%%#-.:%%-.=%%-................
    ...............=%%-.::+%#..*%#-*%#-...=%%%%%%%%:..%%-.#%%%=:%%=..*%#---:...#%%%%%%%*................
    ...............:*%%*=+%%#..*%#:.*%%:.:#%*...:#%*..%%-.-%%+.:#%=..*%*......=%%-...=%%-...............
    .................-**#**-...+*+:..+**.-**:....=**:.**:.......**=..=*+......+*+.....**+...............
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    """


if __name__ == "__main__":
    # print('GRAMPA is initializing...')
    print(welcome_ascii())

    if int(sys.version[0]) < 3:
        sys.exit("PLEASE USE PYTHON3 TO RUN THIS CODE. EXITING")

    parser = argparse.ArgumentParser(
        description="Create a magnetic field model with user-specified parameters."
    )

    # General parameters group
    general_group = parser.add_argument_group("General Parameters")
    general_group.add_argument(
        "-sourcename",
        "--sourcename",
        help="Source/Cluster name, for saving purposes.",
        type=str,
        required=True,
    )
    general_group.add_argument(
        "-reffreq",
        "--reffreq",
        help="Observed radiation frequency in MHz (i.e. center of the band).",
        type=float,
        required=True,
    )
    general_group.add_argument(
        "-cz", "--cz", help="Source/Cluster redshift.", type=float, required=True
    )
    general_group.add_argument(
        "-iteration",
        "--iteration",
        help="For saving different random initializations (default 0).",
        type=int,
        default=0,
    )

    # Physical model parameters group
    model_group = parser.add_argument_group("Physical Model Parameters")
    # Magnetic field parameters
    model_group.add_argument(
        "-xi",
        "--xi",
        help="Vector potential spectral index (= 2 + {Bfield power law spectral index}, Default Kolmogorov.)",
        type=float,
        default=5.67,
    )
    model_group.add_argument(
        "-eta",
        "--eta",
        help="Exponent relating B field to electron density profile (default 0.5).",
        type=float,
        default=0.5,
    )
    model_group.add_argument(
        "-B0",
        "--B0",
        help="Central magnetic field strength in muG. (Default 1.0).",
        type=float,
        default=1.0,
    )
    model_group.add_argument(
        "-lambdamax",
        "--lambdamax",
        help="Magnetic field maximum fluctuation scale in kpc. (Default None, i.e. max, i.e. one reversal scale = (max-size of grid)/2 ).",
        default=None,
        type=float,
    )
    model_group.add_argument(
        "-pixsize",
        "--pixsize",
        help="Pixsize in kpc. Default 1 pix = 3 kpc.",
        type=float,
        default=3.0,
    )
    model_group.add_argument(
        "-N",
        "--N",
        help="Amount of pixels (default 512, power of 2 recommended).",
        type=int,
        default=512,
    )
    # Electron density beta model parameters
    model_group.add_argument(
        "-ne0",
        "--ne0",
        help="Central electron density for beta model in cm^-3.",
        type=float,
        default=0.0031,
    )
    model_group.add_argument(
        "-rc", "--rc", help="Core radius in kpc.", type=float, default=341
    )
    model_group.add_argument(
        "-beta", "--beta", help="Beta power for beta model.", type=float, default=0.77
    )
    # If we want to add fluctuations to the electron density
    model_group.add_argument(
        "-fluctuate_ne",
        "--fluctuate_ne",
        help="Whether to add lognormal fluctuations to the electron density. Default False",
        type=str2bool,
        default=False,
    )
    model_group.add_argument(
        "-mu_ne_fluct",
        "--mu_ne_fluct",
        help="Mean of the fluctuations in the electron density. Is not important because we normalise with the electron density profile anyway.",
        type=float,
        default=1.0,
    )
    model_group.add_argument(
        "-sigma_ne_fluct",
        "--sigma_ne_fluct",
        help="Standard deviation of the fluctuations in the electron density",
        type=float,
        default=0.2,
    )
    model_group.add_argument(
        "-lambdamax_ne_fluct",
        "--lambdamax_ne_fluct",
        help="Electron density maximum fluctuation scale in kpc. (Default None, i.e. one reversal scale: max size of grid/2).",
        default=None,
        type=float,
    )

    # Computational parameters group
    computational_group = parser.add_argument_group("Computational Parameters")
    computational_group.add_argument(
        "-dtype",
        "--dtype",
        help="Bit type to use 32 bit (default) or 64 bit.",
        type=int,
        default=32,
    )
    computational_group.add_argument(
        "-garbagecollect",
        "--garbagecollect",
        help="Let script manually free up memory in key places (default True).",
        type=str2bool,
        default=True,
    )
    computational_group.add_argument(
        "-recompute",
        "--recompute",
        help="Whether to recompute even if data already exists. (default False).",
        type=str2bool,
        default=False,
    )
    computational_group.add_argument(
        "-subcube_ne",
        "--subcube_ne",
        help="Whether to assume electron density profile is spherically symmetrical and speed up calculations. (default True)",
        type=str2bool,
        default=True,
    )

    # Output parameters group
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "-savedir",
        "--savedir",
        help='Where to save results. (default current working dir "./")',
        type=str,
        default="./",
    )
    output_group.add_argument(
        "-saverawfields",
        "--saverawfields",
        help="Whether to save the unnormalised A vector potential and B field. (default True)",
        type=str2bool,
        default=True,
    )
    output_group.add_argument(
        "-saveresults",
        "--saveresults",
        help="Whether to save the normalised B field, RM images, etc (default True).",
        type=str2bool,
        default=True,
    )

    # Imaging parameters group
    imaging_group = parser.add_argument_group("Imaging Parameters")
    imaging_group.add_argument(
        "-beamsize",
        "--beamsize",
        help="Image beam size in arcsec, for smoothing RM images and depolarisation (default 20asec).",
        type=float,
        default=20.0,
    )
    imaging_group.add_argument(
        "-frame",
        "--frame",
        choices=["observedframe", "clusterframe"],
        help="Which frame to compute the RM and depol images. Default = observedframe.",
        default="observedframe",
    )

    # Testing and recomputation group
    testing_group = parser.add_argument_group("Testing and Re-computation")
    testing_group.add_argument(
        "-testing",
        "--testing",
        help="Produce validation plots (default False).",
        type=str2bool,
        default=False,
    )

    args = vars(parser.parse_args())

    # The electron density model (can replace by own model, currently wraps around beta model)
    def ne_funct(r):
        return mutils.beta_model(r, args["ne0"], args["rc"], args["beta"])

    # Initialise the model with arguments and density function
    model = MagneticFieldModel(args, ne_funct=ne_funct)

    # Start the actual calculation
    model.run_model()

    """
    # for testing
    self = model
    self.run_model()

    rm *log
    run grampa/magneticfieldmodel.py -sourcename test -reffreq 944 -xi 5.67 -N 256 -pixsize 10.0 -eta 0.5 -B0 1 -dtype 32 -beamsize 20 -recompute True -savedir ../tests_local/ -saverawfields True -saveresults True -cz 0.021 -testing True -fluctuate_ne False
    python grampa/magneticfieldmodel.py -sourcename test -reffreq 944 -xi 5.67 -N 256 -pixsize 10.0 -eta 0.5 -B0 1 -dtype 32 -beamsize 20 -recompute True -savedir ../tests_local/ -saverawfields True -saveresults True -cz 0.021 -testing True -fluctuate_ne True -sigma_ne_fluct 0.2 -lambdamax_ne_fluct 100
    """
