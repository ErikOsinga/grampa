import os
import sys
import numpy as np
import pyfftw

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.constants import c as speed_of_light

import argparse
import time
import gc 
import psutil
import logging

import magneticfieldmodel_utils as mutils

cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # TODO, make optional

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

__version__ = '0.0.1'



class MagneticFieldModel:
    def __init__(self, args):
        self.starttime = time.time()
        pid = os.getpid()
        self.python_process = psutil.Process(pid)
        
        self.logger = logging.getLogger(__name__)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.logfile = f'grampa_logs_{timestr}.log'

        # File handler
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s'))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s'))
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Set level
        self.logger.setLevel(logging.INFO)
        self.logger.info('Creating magnetic field model with GRAMPA...')

        self.sourcename = args['sourcename']
        self.reffreq = args['reffreq']
        self.cz = args['cz']
        self.N = args['N']
        self.pixsize = args['pixsize']
        self.xi = args['xi']
        self.eta = args['eta']
        self.B0 = args['B0']
        self.lambdamax = args['lambdamax'] # user input lambdamax, can be None
        self.check_lambdamax() # convert lambdamax to a number in kpc if None
        self.dtype = args['dtype']
        self.check_ftype_ctype()
        self.garbagecollect = args['garbagecollect']
        self.iteration = args['iteration']
        self.beamsize = args['beamsize']
        self.recompute = args['recompute']
        self.ne0 = args['ne0']
        self.rc = args['rc']
        self.beta = args['beta']
        self.testing = args['testing']
        self.savedir = args['savedir']
        self.saverawfields = args['saverawfields']
        self.saveresults = args['saveresults']
        self.redshift_dilution = args['redshift_dilution']
        self.nthreads_fft = 48

        # Check resolution and beam size
        self.check_resolution_beamsize()

        # Create paramstring for saving files
        self.paramstring = self.create_paramstring()

        # Check directory where outputs will be saved
        self.check_savedir()

        # log parameters to file
        self.log_parameters()

    def check_lambdamax(self):
        if self.lambdamax is None:
            # Smallest possible k mode (k=1) corresponds to Lambda=(N*pixsize)/2, one reversal
            self.lambdamax =(self.N*self.pixsize)/2
        elif self.lambdamax > (self.N*self.pixsize)/2:
            self.lambdamax =(self.N*self.pixsize)/2
            self.logger.warning(f"Warning: Input Lambda_max is larger than the maximum possible scale. Setting Lambda_max to maximum possible scale of {self.lambdamax} kpc.")
        elif self.lambdamax < 0:
            errormsg = f"Error: {self.lambdamax=} but it cannot be negative."
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        return
    
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
            errormsg = f"Cannot set dtype to float{self.dtype}. Valid values are 32 or 64."
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        return

    def check_resolution_beamsize(self):
        # Calculate the resolution we have at cluster redshift with an X arcsec beam
        resolution = (cosmo.kpc_proper_per_arcmin(self.cz) * self.beamsize * u.arcsec).to(u.kpc)
        self.FWHM = resolution.value  # in kpc

        if self.FWHM < self.pixsize * 5:  # 5 is approximately 2 * 2sqrt(2ln(2))  (because we want at least 2 pix)
            # Set it automatically so the FWHM corresponds to 5 * pixsize at cluster redshift
            self.logger.warning(f"User input angular resolution of {self.beamsize} arcsec corresponds to physical resolution of {self.FWHM:.2f} kpc (FWHM).")
            self.FWHM = self.pixsize * 5  # kpc
            self.beamsize = (self.FWHM * u.kpc / (cosmo.kpc_proper_per_arcmin(self.cz).to(u.kpc / u.arcsec))).to(u.arcsec).value
            self.logger.warning(f"WARNING: However, models are being ran with p={self.pixsize} kpc. The code will smooth to {self.FWHM} kpc automatically. This corresponds to a beam size of {self.beamsize:.2f} arcsec instead. Please keep this in mind.")
        return 

    def create_paramstring(self):
        if self.lambdamax is None:
            paramstring = 'N=%i_p=%i_B0=%.1f_%s_eta=%.2f_s=%s%s%s' % (
                self.N, self.pixsize, self.B0, 'xi=%.2f' % self.xi, self.eta, self.sourcename, '_it' + str(self.iteration), '_b%.2fasec' % self.beamsize)
        else:
            paramstring = 'N=%i_p=%i_B0=%.1f_%s_eta=%.2f_s=%s_Lmax=%i%s%s' % (
                self.N, self.pixsize, self.B0, 'xi=%.2f' % self.xi, self.eta, self.sourcename, self.lambdamax, '_it' + str(self.iteration), '_b%.2fasec' % self.beamsize)
        if self.redshift_dilution:
            paramstring += '_zd'
        return paramstring
    
    def log_parameters(self):
        self.logger.info("Using parameters:")
        self.logger.info(f" xi={self.xi:.2f} (n={self.xi-2:.2f})")
        self.logger.info(f" N={self.N}")
        self.logger.info(f" eta={self.eta:.1f}")
        self.logger.info(f" B0={self.B0:.1f}")
        self.logger.info(f" pixsize={self.pixsize:.1f}")
        self.logger.info(f" sourcename= {self.sourcename}")
        self.logger.info(f" cz= {self.cz:.2f}")
        self.logger.info(f" Lambda_max= {self.lambdamax}")
        self.logger.info(f" Beam FWHM = {self.beamsize:.1f} arcsec")
        self.logger.info(f" Beam FWHM = {self.FWHM:.1f} kpc")
        self.logger.info(f" dtype= float{self.dtype}")
        self.logger.info(f" Manual garbagecollect= {self.garbagecollect}")
        self.logger.info(f" ne0= {self.ne0:.2f}")
        self.logger.info(f" rc= {self.rc:.2f}")
        self.logger.info(f" beta= {self.beta:.2f}")
        self.logger.info(f" testing= {self.testing}")

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
        if os.path.isfile(f"{savedir2}RMimage_{self.paramstring}.npy") and os.path.isfile(f"{savedir2}RMhalfconvolved_{self.paramstring}.npy"):
            return 'fully computed'
        else:
            # Check if the result with B0=1 is already computed. We can use it
            # to compute the result with any other B0
            if os.path.isfile(f"{savedir2}RMimage_{self.paramstring}.npy") and os.path.isfile(f"{savedir2}RMhalfconvolved_{self.paramstring}.npy"):
                return 'partially computed'
            else:
                return 'not computed'

    def run_model(self):
        """
        Where the magic happens
        """

        # First check whether the results are already computed
        status = self.check_results_already_computed()

        if not self.recompute and status == 'fully computed':
            dtime = time.time()-self.starttime
            self.logger.info(f"Script fully finished. Took {dtime:%.1f} seconds to check results")
            self.logger.info("Results already computed and recompute=False, exiting.")
            sys.exit("Results already computed and recompute=False, exiting.")

        # Otherwise status = partially computed or not computed, continue.
        if not self.recompute and status == 'partially computed':
            self.logger.info(f"Loading RM image from file with B0=1, and scaling it to B0={self.B0}") # todo, load any field strength
            RMimage, RMimage_half, RMconvolved, RMhalfconvolved = self.computeRMimage_from_file()

        else: # otherwise we need to compute the RM images, could be from scratch or from vectorpotential or Bfield
            
            # Check whether the vector potential file or Bfield file already exists, would save time
            already_computed_Afield, already_computed_Bfield, vectorpotential_file, Bfield_file = self.check_results_computed()

            if self.recompute:
                self.logger.info("User forces recompute, so recomputing everything.")
                already_computed_Afield = False
                already_computed_Bfield = False

            if already_computed_Bfield:
                self.logger.info("Found a saved version of the (pre-normalisation) magnetic field with user defined parameters.")
                self.logger.info(f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}")
                self.logger.info("Loading from file..")
                B_field = np.load(Bfield_file)
            
            elif already_computed_Afield:
                self.logger.info("Found a saved version of the vector potential with user defined parameters.")
                self.logger.info(f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}")
                self.logger.info("Loading from file..")
                
                # then compute B from A
                B_field = self.computeBfromA(vectorpotential_file)

            else:
                if not self.recompute: 
                    self.logger.info("No saved version of the magnetic field or vector potential found.")    
                self.logger.info("Starting from scratch...")

                # compute the vector potential
                field = mutils.calculate_vectorpotential(self.N, self.xi, self.lambdamax, self.pixsize, self.ftype)
                
                # compute B from A
                B_field = self.computeBfromA(None, field)

            # If they were not already computed, we'll save them to file for future use
            if not already_computed_Afield and self.saverawfields:
                self.logger.info(f"Saving fourier vector potential to {vectorpotential_file}, such that it can be used again")
                np.save(vectorpotential_file, field)
            if not already_computed_Bfield and self.saverawfields:
                self.logger.info(f"Saving unnormalised magnetic field to {Bfield_file}, such that it can be used again")
                np.save(Bfield_file, B_field)

                self.logger.debug(f"Resulting magnetic field shape: {B_field.shape}")


            ########## Normalisation of the B field with some density profile ##########
            self.logger.info("Normalising profile with electron density profile")

            ## Using radial symmetry in a way where we can only use 1/8th of the cube
            ## we can calculate ne_3d about 6x faster for N=1024
            subcube = False # TODO, change when lognormal is implemented
            self.logger.info(f"Using subcube symmetry to speed up calculations: {subcube}")

            # Vector denoting the real space positions. The 0 point is in the middle.
            # Now runs from -31 to +32 which is 64 values. Or 0 to +32 when subcube=True
            # The norm of the position vector
            xvec_length = mutils.xvector_length(self.N, 3, self.pixsize, self.ftype, subcube=subcube)

            # 3d cube of electron density
            ne_3d = ne_funct(xvec_length, self.ne0, self.rc, self.beta)

            del xvec_length # We dont need xvec_length anymore

            if subcube:
                c = 0 # then the center pixel is the first one, because the subcube is only the positive subset
            else:
                # Make sure n_e is not infinite in the center. Just set it to the pixel next to it
                c = self.N//2-1

            # Make sure n_e is not infinite in the center. Just set it to the pixel next to it
            ne_3d[c,c,c] = ne_3d[c,c+1,c]
            ne0 = ne_3d[c,c,c] # Electron density in center of cluster

            # Normalise the B field such that it follows the electron density profile ^eta
            B_field_norm, ne_3d = mutils.normalise_Bfield(ne_3d, ne0, B_field, self.eta, self.B0, subcube)
            del B_field # We dont need B field unnormalised anymore
            if self.garbagecollect:
                self.logger.info("Deleted B_field and xvec_length. Collecting garbage..")
                gc.collect()
                memoryUse = self.python_process.memory_info()[0]/2.**30
                self.logger.info(f'Memory used: {memoryUse:.1f} GB')

            # Save the B_field_norm as a class variable
            self.B_field_norm = B_field_norm

            # Calculate the B_field amplitude (length of the vector)
            # B_field_amplitude_nonorm = np.copy(B_field_amplitude)
            # B_field_amplitude = np.linalg.norm(B_field_norm,axis=3)

            if self.testing:
                print("Plotting normalised B-field amplitude")
                dens_func = lambda r: ne_funct(r, self.ne0, self.rc, self.beta)  # noqa: E731
                mutils.plot_Bfield_amp_vs_radius(B_field_norm, self.pixsize, dens_func, self.B0)
                print("Plotting normalised B-field power spectrum")
                mutils.plot_B_field_powerspectrum(B_field_norm, self.xi, self.lambdamax)

            print ("Calculating rotation measure images.")
            # now we need full 3D density cube
            if subcube:
                ne_3d = mutils.cube_from_subcube(ne_3d, self.N, self.ftype)

            # Calculate the RM by integrating over the 3rd axis
            RMimage = mutils.RM_integration(ne_3d,B_field_norm,self.pixsize,axis=2)
            # Also integrate over half of the third axis. For in-cluster sources
            RMimage_half = mutils.RM_halfway(ne_3d,B_field_norm,self.pixsize,axis=2)

            # Convolve the RM image with this resolution.
            # From here we can start to use float64 again, because the images are 2D
            RMconvolved, RMhalfconvolved = mutils.convolve_with_beam([RMimage,RMimage_half], self.FWHM, self.pixsize)

            # Save the RM images as class variables
            self.RMimage = RMimage
            self.RMimage_half = RMimage_half
            self.RMconvolved = RMconvolved
            self.RMhalfconvolved = RMhalfconvolved

        # Now we have the RM images either from rescaling or from scratch
        if self.testing:
            print("Plotting RM images. Unconvolved & convolved")    
            mutils.plotRMimage(RMimage, self.pixsize, title='RM image not convolved')
            mutils.plotRMimage(RMconvolved, self.pixsize, title='RM image convolved')
            print("Plotting RM power spectrum")
            mutils.plot_RM_powerspectrum(RMimage, self.xi, self.lambdamax, title='RM image not convolved')
            mutils.plot_RM_powerspectrum(RMconvolved, self.xi, self.lambdamax, title='RM image convolved')

        # Calculate Stokes Q and U images at the centre frequency
        # rotate their polarisation angle with the RM map, and convolve them
        # to produce beam depolarisation

        # Randomly set an intrinsic polarisation angle (uniform)
        phi_intrinsic = 45*np.pi/180 # degrees to radians
        #Observed wavelength of radiation in meters
        wavelength = (speed_of_light/(self.reffreq*u.MHz)).to(u.m).value

    def computeRMimage_from_file(self):
        """
        If we already have an RM image with B0=1, we can simply scale it to any other B0
        because we're simply doing  X * integral(B*ne) dr = X * RM
        """
        savedir2 = self.savedir + f'after_normalise/{self.sourcename}/'
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

    def computeBfromA(self, vectorpotential_file, field=None):
        """
        Compute the magnetic field from the vector potential. Cross product of k and A
        """
        if field is None:
            # Load vector potential file
            field = np.load(vectorpotential_file)
            
        self.logger.info(f"Generating k vector in ({self.N},{self.N},{self.N//2},3) space")
        kvec = mutils.kvector(self.N, 3, self.pixsize, self.ftype) # 3 = number of dimensions of the vector field
        
        # Fourier B field = Cross product  B = ik \cross A 
        self.logger.info("Calculating magnetic field using the crossproduct Equation")
        field = mutils.magnetic_field_crossproduct(kvec, field, self.N, self.ctype)
        del kvec # Huge array which we dont need anymore 
        if self.garbagecollect: 
            self.logger.info("Deleted kvec. Collecting garbage..")
            gc.collect()
            memoryUse = self.python_process.memory_info()[0]/2.**30
            self.logger.info(f'Memory used: {memoryUse:.1f} GB')

        # B field is the inverse fourier transform of fourier_B_field
        run_ift = pyfftw.builders.irfftn(field,s=(self.N,self.N,self.N),axes=(0,1,2)
            , auto_contiguous=False, auto_align_input=False, avoid_copy=True,threads=self.nthreads_fft)
        field = run_ift()
        # memoryUse = self.python_process.memory_info()[0]/2.**30
        # self.logger.info('Memory used: %.1f GB'%memoryUse)
        if self.garbagecollect: 
            self.logger.info("Ran IFFT.. Collecting garbage..")
            gc.collect()
            memoryUse = self.python_process.memory_info()[0]/2.**30
            self.logger.info(f'Memory used: {memoryUse:.1f} GB')

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
            self.logger.info("Found a saved version of the vector potential with user defined parameters:")
            self.logger.info(f" N={self.N} xi={self.xi:.2f} Lmax={self.lambdamax}, pixsize={self.pixsize}")

            self.logger.info("Checking if magnetic field was also already computed..")
            if os.path.isfile(Bfield_file):
                self.logger.info("Magnetic field already computed in a previous run")
                already_computed_Bfield = True
            else:
                self.logger.info("Magnetic field not computed in a previous run")
        
        return already_computed_Afield, already_computed_Bfield, vectorpotential_file, Bfield_file

    def BandAfieldfiles(self):
        # TODO: if lambda_max is the maximum scale of the grid
        # then the power spectrum is scale invariant, so pixel size is not important
        # could save the B and A field files without pixsize in the name

        vectorpotential_file = f"{self.savedir}Afield_N={self.N:.0f}_p={self.pixsize:.0f}_xi={self.xi:.2f}_Lmax={self.lambdamax:.0f}_it{self.iteration}.npy"
        Bfield_file = f"{self.savedir}Bfield_N={self.N:.0f}_p={self.pixsize:.0f}_xi={self.xi:.2f}_Lmax={self.lambdamax:.0f}_it{self.iteration}.npy"
        
        return vectorpotential_file, Bfield_file

# The electron density model (can replace by own model, currently wraps around beta model)
def ne_funct(r, ne0, rc, beta):
    return mutils.beta_model(r, ne0, rc, beta)

if __name__ == "__main__":
    if int(sys.version[0]) < 3:
        sys.exit("PLEASE USE PYTHON3 TO RUN THIS CODE. EXITING")    

    parser = argparse.ArgumentParser(description='Create a magnetic field model with user specified parameters')
    parser.add_argument('-sourcename','--sourcename', help='Cluster name, for saving purposes', type=str, required=True)
    parser.add_argument('-reffreq','--reffreq', help='Reference Frequency in MHz (i.e. center of the band)', type=float, required=True)
    parser.add_argument('-cz','--cz', help='Cluster redshift', type=float, required=True)
    parser.add_argument('-xi','--xi', help='Vector potential spectral index (= 2 + {Bfield power law spectral index}, default Kolmogorov )', type=float, default=5.67)
    parser.add_argument('-N' ,'--N', help='Amount of pixels (default 512, power of 2 recommended)', type=int, default=512)
    parser.add_argument('-pixsize','--pixsize', help='Pixsize in kpc. Default 1 pix = 3 kpc.', default=3.0, type=float)
    parser.add_argument('-eta','--eta', help='Exponent relating B field to electron density profile, default 0.5', type=float, default=0.5)
    parser.add_argument('-B0','--B0', help='Central magnetic field strength in muG. Default 1.0', type=float, default=1.0)
    parser.add_argument('-lmax','--lambdamax', help='Maximum scale in kpc. Default None (i.e. max size of grid/2).', default=None, type=float)
    parser.add_argument('-dtype','--dtype', help='Float type to use 32 bit (default) or 64 bit', default=32, type=int)
    parser.add_argument('-garbagecollect','--garbagecollect', help='Let script manually free up memory (Default True)', default=True, type=bool)
    ## todo
    # parser.add_argument('-doUPP','--doUPP', help='If X-ray is not found, continue with UPP', default=False, type=bool)
    parser.add_argument('-iteration' ,'--iteration', help='For saving different random initializations. Default 0', type=int, default=0)
    parser.add_argument('-beamsize' ,'--beamsize', help='Image beam size in arcsec, for smoothing. Default 20asec', type=float, default=20.0)
    parser.add_argument('-recompute' ,'--recompute', help='Whether to recompute even if data already exists. Default False', type=bool, default=False)
    parser.add_argument('-ne0' ,'--ne0', help='Central electron density in beta model.', type=float, default=0.0031)
    parser.add_argument('-rc' ,'--rc', help='Core radius in kpc', type=float, default=341)
    parser.add_argument('-beta' ,'--beta', help='Beta power for beta model', type=float, default=0.77)
    parser.add_argument('-testing','--testing', help='Produce validation plots. Default False', default=False, type=bool)

    parser.add_argument('-savedir' ,'--savedir', help='Where to save results. Default ./', type=str, default="./")
    parser.add_argument('-saverawfields','--saverawfields', help='Whether to save the unnormalised A vector potential and B field. These can be quite big, but allow rapid re-calculation of different ne normalisations. Default True', default=True, type=bool)
    parser.add_argument('-saveresults','--saveresults', help='Whether to save the normalised B field, RM images, etc (everything after normalising the B field). Default True', default=True, type=bool)

    parser.add_argument('-redshift_dilution', '--redshift_dilution', help='Calculate the RM in the cluster frame (True) or in observed frame (False)', default=True, type=bool)

    args = vars(parser.parse_args()) 

    # Start the actual calculation
    model = MagneticFieldModel(args)

    # for testing
    self = model

    self.run_model()
    # model.run_model()

    """
    # for testing
    run magneticfieldmodel.py -sourcename test -reffreq 944 -xi 5.67 -N 256 -pixsize 3.0 -eta 0.5 -B0 1 -dtype 32 -beamsize 20 -recompute True -savedir ../tests_local/ -saverawfields False -saveresults True -cz 0.021 -testing True
    python magneticfieldmodel.py -sourcename test -reffreq 944 -xi 5.67 -N 256 -pixsize 3.0 -eta 0.5 -B0 1 -dtype 32 -beamsize 20 -recompute True -savedir ../tests_local/ -saverawfields False -saveresults True -cz 0.021 -testing True
    """
