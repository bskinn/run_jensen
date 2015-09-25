#-------------------------------------------------------------------------------
# Name:        run_jensen
# Purpose:      Automated execution of diatomic M-L computations in the form
#               of Jensen et al. J Chem Phys 126, 014103 (2007)
#
# Author:      Brian
#
# Created:     8 May 2015
# Copyright:   (c) Brian 2015
# License:     The MIT License; see "license.txt" for full license terms
#                   and contributor agreement.
#
#       This file is a standalone module for execution of M-L diatomic
#           computations in ORCA, approximately per the approach given in
#           the above citation.
#
#       http://www.github.com/bskinn/run_jensen
#
#-------------------------------------------------------------------------------

# Module-level imports
import os, logging, time, re, csv
import h5py as h5, numpy as np


# Module-level variables
# Constant strings
repofname = 'jensen.h5'
csvfname = 'jensen.csv'
csv_multfname = 'jensen_mult.csv'
pausefname = 'pause'
dir_fmt = 'jensen_%Y%m%d_%H%M%S'
sep = "_"
NA_str = "NA"

# Adjustable parameters (not all actually are adjustable yet)
exec_cmd = 'runorca_pal.bat'
opt_str = '! TIGHTOPT'
convergers = ["", "! KDIIS", "! SOSCF"] #, "! NRSCF"]
req_tag_strs = ['<MOREAD>', '<MULT>', '<XYZ>', '<OPT>', '<CONV>', \
                '<CHARGE>']
init_dia_sep = 2.1  # Angstroms
fixed_dia_sep = True
ditch_sep_thresh = 4.0
geom_scale = 0.75
pausetime = 2.0
skip_atoms = False

# Class for logging information
class log_names(object):
    filename = 'jensen_log.txt'
    fmt = "%(levelname)-8s [%(asctime)s] %(message)s"
    datefmt = "%H:%M:%S"
    loggername = 'rjLogger'
    handlername = 'rjHandler'
    formattername = 'rjFormatter'
## end class log_names

# Class for names of subgroups
class h5_names(object):
    max_mult = 'max_mult'
    mult_prfx = 'm'
    chg_prfx = 'q'
    ref_prfx = 'r'
    run_base = 'base'
    converger = 'conv'
    out_en = 'energy'
    out_zpe = 'zpe'
    out_enth = 'enthalpy'
    out_bondlen = 'bond_length'
    out_dipmom = 'dipole_moment'
    min_en = 'min_en'
    min_en_mult = 'min_en_mult'
    min_en_ref = 'min_en_ref'
    min_en_zpe = 'min_en_zpe'
    min_en_enth = 'min_en_enth'
    min_en_bondlen = 'min_en_bondlen'
    min_en_dipmom = 'min_en_dipmom'
## end class h5_names

# Regex patterns for quick testing
p_multgrp = re.compile("/" + h5_names.mult_prfx + "(?P<mult>[0-9]+)$")
p_refgrp = re.compile("/" + h5_names.ref_prfx + "(?P<ref>[0-9]+)$")

# Atomic numbers for elements, and the max associated unpaired electrons
metals = set(range(21,31))
nonmetals = set([1, 6, 7, 8, 9, 16, 17, 35])
cation_nms = set([1,8])
max_unpaired = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 16: 4, 17: 1, 35: 1, \
                21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 6, 27: 5, \
                28: 4, 29: 3, 30: 2}
mult_range = range(1,13)


def do_run(template_file, wkdir=None):
    """ Top-level function for doing a series of runs

    """

    # Imports
    from opan.utils import make_timestamp
    from opan.const import atomSym

    # If wkdir specified, try changing there first
    if not wkdir == None:
        old_wkdir = os.getcwd()
        os.chdir(wkdir)
    ## end if

    # Pull in the template
    with open(template_file) as f:
        template_str = f.read()
    ## end with

    # Create working folder, enter
    dir_name = time.strftime(dir_fmt)
    os.mkdir(dir_name)
    os.chdir(dir_name)

    # Set up and create the log, and log wkdir
    setup_logger()
    logger = logging.getLogger(log_names.loggername)
    logger.info("Jensen calc series started: " + time.strftime("%c"))
    logger.info("Working in directory: " + os.getcwd())

    # Proofread the template
    proof_template(template_str)

    # Log the template file contents
    logger.info("Template file '" + template_file + "' contents:\n\n" + \
                                                                template_str)

    # Log the metals and nonmetals to be processed, including those
    #  nonmetals for which the monocations will be calculated.
    logger.info("Metals: " + ", ".join([atomSym[a].capitalize() \
                                                    for a in metals]))
    logger.info("Non-metals: " + ", ".join([atomSym[a].capitalize() \
                                                    for a in nonmetals]))
    logger.info("Cations calculated for non-metals: " + \
                    ", ".join([atomSym[a].capitalize() for a in cation_nms]))

    # Log the geometry scale-down factor, if used
    if fixed_dia_sep:
        logger.info("Using fixed initial diatomic separation of " + \
                str(init_dia_sep) + " Angstroms.")
    else:
        logger.info("Using geometry scale-down factor: " + str(geom_scale))
    ## end if

    # Store the starting time
    start_time = time.time()

    # Create the data repository
    repo = h5.File(repofname, 'a')

    # Log notice if skipping atoms
    if skip_atoms:
        logger.warning("SKIPPING ATOM COMPUTATIONS")
    else:
        # Loop atoms (atomic calculations)
        for at in metals.union(nonmetals):
            run_mono(at, template_str, repo)
            repo.flush()
        ## next at
    ## end if

    # Loop atom pairs (diatomics) for run execution
    for m in metals:
        for nm in nonmetals:
            # Run the diatomic optimizations
            run_dia(m, nm, 0, template_str, repo)

            # Ensure repository is updated
            repo.flush()

            # Run the diatomic monocation optimizations for hydrides, oxides
            if nm in cation_nms:
                run_dia(m, nm, 1, template_str, repo)
            ## end if

            # Ensure repository is updated
            repo.flush()

            # Clear any residual temp files from failed comps
            clear_tmp(atomSym[m].capitalize() + atomSym[nm].capitalize())

        ## next nm
    ## next nm

    # Generate the results csv
    write_csv(repo)
    logger.info("CSV file generated.")

    # Close the repository
    repo.close()

    # Exit the working directory; if wkdir not specified then just go to
    #  parent directory; otherwise restore the old wd.
    if wkdir == None:
        os.chdir('..')
    else:
        os.chdir(old_wkdir)
    ## end if

    # Log end of execution
    logger.info("Calc series ended: " + time.strftime("%c"))
    logger.info("Total elapsed time: " + \
                                    make_timestamp(time.time() - start_time))

## end def do_run


def continue_dia(template_file, m, nm, chg, mult, ref, wkdir=None):

    # Imports
    from opan.utils import make_timestamp
    from opan.const import atomSym

    # If wkdir specified, try changing there first
    if not wkdir == None:
        old_wkdir = os.getcwd()
        os.chdir(wkdir)
    ## end if

    # Pull in the template
    with open(template_file) as f:
        template_str = f.read()
    ## end with

    # Set up and create the log, and log wkdir
    setup_logger()
    logger = logging.getLogger(log_names.loggername)
    logger.info("============  RESTART  ============")
    logger.info("Jensen calc series resumed: " + time.strftime("%c"))
    logger.info("Working in directory: " + os.getcwd())

    # Proofread the template
    proof_template(template_str)

    # Log the template file contents
    logger.info("Template file '" + template_file + "' contents:\n\n" + \
                                                                template_str)

    # Store the starting time
    start_time = time.time()

    # Retrieve the data repository
    repo = h5.File(repofname, 'a')

    # Log the restart
    logger.info("Restarting '" + build_base(m, nm, chg) + \
                "' at multiplicity " + str(mult) + ", reference " + \
                "multiplicity " + str(ref))

    # Run the diatomic optimizations, with restart point
    run_dia(m, nm, chg, template_str, repo, startvals=(mult, ref))

    # Ensure repository is updated
    repo.flush()

    # Clear any residual temp files from failed comps
    clear_tmp(atomSym[m].capitalize() + atomSym[nm].capitalize())

    # Generate the results csv
    write_csv(repo)
    logger.info("CSV file generated.")

    # Close the repository
    repo.close()

    # If workdir specified, return to old
    if wkdir != None:
        os.chdir(old_wkdir)
    ## end if

    # Log end of execution
    logger.info("Diatomic recalc ended: " + time.strftime("%c"))
    logger.info("Total elapsed time: " + \
                                    make_timestamp(time.time() - start_time))

## end def continue_dia


def recover_results():
    """ #DOC: Docstring for recover_results
    """

    # Imports

    # Bind repo
    repo = h5.File(repofname)



## end def recover_results

def write_csv(repo):
    """ #DOC: Docstring for write_csv
        Is split into its own method to enable re-running later on
    """

    # Imports
    from opan.const import atomSym

    # Generate the csv
    with open(csvfname, mode='w') as csv_f:
        # Create the CSV writer object
        w = csv.writer(csv_f, dialect='excel')

        # Write a header line
        w.writerow(["M", "NM", "E_M", "H_M", "E_NM", "H_NM", "E_dia", "H_dia", \
                "ZPE_dia", \
                "E_ion", "H_ion", "ZPE_ion", "Mult_0", "Mult_1", "BDE", "IE", \
                "Bond_0", "Bond_1", \
                "Dipole_0", "Dipole_1"])

        for m in metals:
            # Link the metal repo group and helper name string
            m_name = atomSym[m].capitalize()
            m_gp = repo.get(m_name)     # Stores None if group not present.

            for nm in nonmetals:
                # Helper name strings
                nm_name = atomSym[nm].capitalize()
                dia_name = m_name + nm_name

                # Group links; all store None if group is not present.
                nm_gp = repo.get(nm_name)
                dia_gp = repo.get(build_base(m, nm, 0))
                ion_gp = repo.get(build_base(m, nm, 1))

                # Pull data, if present, and calculate results
                # Energies
                en_m = safe_retr(m_gp, h5_names.min_en, \
                                m_name + " energy")
                enth_m = safe_retr(m_gp, h5_names.min_en_enth, \
                                m_name + " enthalpy correction")
                en_nm = safe_retr(nm_gp, h5_names.min_en, \
                                nm_name + " energy")
                enth_nm = safe_retr(nm_gp, h5_names.min_en_enth, \
                                nm_name + " enthalpy correction")
                en_dia = safe_retr(dia_gp, h5_names.min_en, \
                                dia_name + " energy")
                enth_dia = safe_retr(dia_gp, h5_names.min_en_enth, \
                                dia_name + " enthalpy correction")
                zpe_dia = safe_retr(dia_gp, h5_names.min_en_zpe, \
                                dia_name + " ZPE correction")

                # Calculate the bond dissociation energy
                try:
                    bde = (en_m + enth_m + en_nm + enth_nm) - \
                                                (en_dia + enth_dia + zpe_dia)
                except TypeError:
                    bde = NA_str
                ## end try

                # Store flag for logging absent values (suppress logging for
                #  nonmetals that are not supposed to have cations calc-ed).
                lgabs = nm in cation_nms

                # Retrieve the energies of the monocation diatomic
                en_ion = safe_retr(ion_gp, h5_names.min_en, \
                                        dia_name + " monocation energy", \
                                        log_absent=lgabs)
                enth_ion = safe_retr(ion_gp, h5_names.min_en_enth, \
                            dia_name + " monocation enthalpy correction", \
                            log_absent=lgabs)
                zpe_ion = safe_retr(ion_gp, h5_names.min_en_zpe, \
                            dia_name + " monocation ZPE correction", \
                            log_absent=lgabs)

                # Calculate the ionization energy
                try:
                    ie = (en_ion + enth_ion + zpe_ion) - \
                                    (en_dia + enth_dia + zpe_dia)
                except TypeError:
                    ie = NA_str
                ## end try

                # Get the bond length, dipole moment, and ground spin state
                min_mult_0 = safe_retr(dia_gp, h5_names.min_en_mult, \
                            dia_name + " ground multiplicity")
                bondlen_0 = safe_retr(dia_gp, h5_names.min_en_bondlen, \
                            dia_name + " bond length")
                dipmom_0 = safe_retr(dia_gp, h5_names.min_en_dipmom, \
                            dia_name + " dipole moment")
                min_mult_1 = safe_retr(ion_gp, h5_names.min_en_mult, \
                            dia_name + " cation ground multiplicity", \
                            log_absent=lgabs)
                bondlen_1 = safe_retr(ion_gp, h5_names.min_en_bondlen, \
                            dia_name + " cation bond length", \
                            log_absent=lgabs)
                dipmom_1 = safe_retr(ion_gp, h5_names.min_en_dipmom, \
                            dia_name + " cation dipole moment", \
                            log_absent=lgabs)

                # Write the CSV line
                w.writerow([m_name, nm_name, en_m, enth_m, en_nm, enth_nm, \
                        en_dia, enth_dia, \
                        zpe_dia, en_ion, enth_ion, zpe_ion, min_mult_0, \
                        min_mult_1, \
                        bde, ie, \
                        bondlen_0, bondlen_1, \
                        dipmom_0, dipmom_1])

            ## next nm
        ## next m
    ## end with (closes csv file)

## end def write_csv


def write_mult_csv(repo, absolute=False):
    """ #DOC: Docstring for write_mult_csv
    """

    # Imports
    from opan.const import atomSym

    # Generate the csv
    with open(csv_multfname, mode='w') as csv_f:
        # Create the CSV writer object
        w = csv.writer(csv_f, dialect='excel')

        # Write a header line
        w.writerow(['M', 'NM', 'q'] + [str(i) for i in mult_range])

        # Loop through all available metals; store the name for convenience
        for m in metals:
            m_name = atomSym[m].capitalize()

            # Loop the nonmetals
            for nm in nonmetals:
                # Store the name & diatomic helper
                nm_name = atomSym[nm].capitalize()

                # Loop the two charges
                for q_val in [0,1]:
                    # Set the group
                    wk_gp = repo.get(m_name + nm_name + sep + \
                                            h5_names.chg_prfx + str(q_val))
                    # Check existence
                    if not wk_gp == None:
                        # Exists; reset the storage array and energy reference
                        row_ary = []
                        en_ref = 0.0 if absolute else \
                                (0.0 if (wk_gp.get(h5_names.min_en) == None) \
                                        else wk_gp.get(h5_names.min_en).value)

                        # Store metal, nonmetal and charge
                        map(row_ary.append, [m_name, nm_name, str(q_val)])

                        # Search the full mult range
                        for mult in mult_range:
                            # Try to store the mult group
                            mgp = wk_gp.get(h5_names.mult_prfx + str(mult))

                            # Check if found, and out_en exists
                            if (not mgp == None) and \
                                    (not mgp.get(h5_names.out_en) == None):
                                # If found, tag the energy into the row array
                                row_ary.append( \
                                        str(mgp.get(h5_names.out_en).value - \
                                                                    en_ref))
                            else:
                                # If not, tag empty string
                                row_ary.append('')
                            ## end if
                        ## next mult

                        # Write a csv row for the species
                        w.writerow(row_ary)

                    ## end if wk_gp exists
                ## next q_val
            ## next nm
        ## next m
    ## end with

## end def write_mult_csv


#TODO: Write further CSV writers, perhaps a general one that writes a CSV for
#  a quantity specified by some sort of 'enum'?  Or/also, one that generates
#  summaries of properties across the matrix of multiplicities and reference
#  wavefunctions?


def run_mono(at, template_str, repo):
    """ #DOC: Docstring for run_mono
    """

    # Imports
    from opan.const import atomSym, PHYS
    from opan.utils import execute_orca as exor
    from time import sleep
    import os, logging

    # Retrieve the logger
    logger = logging.getLogger(log_names.loggername)

    # Create/retrieve the group for the atom in the repository
    atgp = repo.require_group(atomSym[at].capitalize())

    # Get the max multiplicity for the atom and store
    max_mult = max_unpaired[at] + 1
    h5_clobber_dataset(atgp, name=h5_names.max_mult, data=max_mult, \
                                                    log_clobber=True)

    # Loop until at minimum possible multiplicity
    for mult in range(max_mult, -(max_mult % 2), -2):
        # Create/retrieve multiplicity subgroup
        mgp = atgp.require_group(h5_names.mult_prfx + str(mult))

        # Define and store run base string
        base = atomSym[at].capitalize() + sep + str(mult)
        h5_clobber_dataset(mgp,name=h5_names.run_base, data=base, \
                                                        log_clobber=True)

        # Try executing, iterating through the convergers
        for conv in convergers:
            # Store converger reporting name
            conv_name = conv if conv != "" else "default"

            # Run the calc, storing the ORCA_OUTPUT object
            oo = exor(template_str, os.getcwd(), [exec_cmd, base], \
                    sim_name=base, \
                    subs=[('OPT', ''), ('CONV', conv), ('MULT', str(mult)), \
                        ('CHARGE', '0'), \
                        ('MOREAD', '! NOAUTOSTART'), \
                        ('XYZ', atomSym[at] + " 0 0 0")])[0]

            # Check if completed and converged
            if oo.completed and oo.converged:
                logger.info(base + " converged using " + conv_name)
                break
            else:
                logger.warning(base + " did not converge using " + conv_name)
            ## end if

            # Run again, with SLOWCONV
            conv_name += " & SLOWCONV"
            oo = exor(template_str, os.getcwd(), [exec_cmd, base], \
                    sim_name=base, \
                    subs=[('OPT', ''), ('CONV', conv + "\n! SLOWCONV"), \
                        ('MULT', str(mult)), ('CHARGE', '0'), \
                        ('MOREAD', '! NOAUTOSTART'), \
                        ('XYZ', atomSym[at] + " 0 0 0")])[0]

            # Again, check if completed / converged
            if oo.completed and oo.converged:
                logger.info(base + " converged using " + conv_name)
                break
            else:
                logger.warning(base + " did not converge using " + conv_name)
            ## end if
        else:
            # If never broken, then the whole series of computations failed.
            #  Log and skip to next multiplicity.
            logger.error(base + " failed to converge in all computations.")
            h5_clobber_dataset(mgp,name=h5_names.converger, \
                                        data="FAILED", log_clobber=True)
            continue  ## to next mult
        ## next conv

        # Store useful outputs
        storage_data = (
                (h5_names.converger, conv_name),
                (h5_names.out_en, oo.en_last()[oo.EN.SCFFINAL]),
                (h5_names.out_zpe, oo.thermo[oo.THERMO.E_ZPE]),
                (h5_names.out_enth, oo.thermo[oo.THERMO.H_IG])
                        )
        for item in storage_data:
            h5_clobber_dataset(mgp,name=item[0], data=item[1], log_clobber=True)
        ## next item
    ## next mult

    # Initialize the minimum energy to zero, which everything should be less
    #  than!  Also init min_mult to ~error value.
    min_en = 0.0
    min_mult = 0

    # Loop over all multiplicity groups to identify the lowest-energy
    #  multiplicity
    for g in atgp.values():
        # Check if a given value is a group and that its name matches the
        #  pattern expected of a multiplicity group
        if isinstance(g, h5.Group) and p_multgrp.search(g.name) != None:
            # Confirm energy value exists
            if g.get(h5_names.out_en) != None:
                # Check if the reported SCF final energy is less than the
                #  current minimum.
                if g.get(h5_names.out_en).value < min_en:
                    # If so, update the minimum and store the multiplicity
                    min_en = np.float_(g.get(h5_names.out_en).value)
                    min_mult = np.int_(p_multgrp.search(g.name).group("mult"))
                ## end if
            ## end if
        ## end if
    ## next g

    # Store the minimum energy and multiplicity, and retrieve/store the other
    #  thermo values
    h5_clobber_dataset(atgp, name=h5_names.min_en, data=min_en, \
                                                        log_clobber=True)
    h5_clobber_dataset(atgp, name=h5_names.min_en_mult, data=min_mult, \
                                                        log_clobber=True)

    # Try to retrieve the appropriate group
    mgp = atgp.get(h5_names.mult_prfx + str(min_mult))

    # If any computation succeeded, *something* will be retrievable. If none
    #  did, mgp will be None; report error if so.
    if mgp == None:
        logger.error("All computations failed for " + \
                atomSym[at].capitalize())
    else:
        h5_clobber_dataset(atgp, name=h5_names.min_en_zpe, \
                    data=mgp.get(h5_names.out_zpe).value, log_clobber=True)
        h5_clobber_dataset(atgp, name=h5_names.min_en_enth, \
                    data=mgp.get(h5_names.out_enth).value, log_clobber=True)
    ## end if

    # Pause if indicator file present
    while os.path.isfile(os.path.join(os.getcwd(),pausefname)):
        sleep(pausetime)
    ## loop

## end def run_mono


def run_dia(m, nm, chg, template_str, repo, startvals=None):
    """ #DOC: Docstring for run_dia
    """

    # Imports
    from opan.const import atomSym, PHYS
    from opan.utils import execute_orca as exor
    from opan.xyz import OPAN_XYZ as XYZ
    import os, logging
    from time import sleep

    # Retrieve logger
    logger = logging.getLogger(log_names.loggername)

    # Create/retrieve group in the repo for the diatomic & charge
    diagp = repo.require_group(build_base(m, nm, chg))

    # Get the max multiplicity for the diatomic and store
    max_mult = max_unpaired[m] + max_unpaired[nm] + 1 - chg
    h5_clobber_dataset(diagp, name=h5_names.max_mult, data=max_mult, \
                                                        log_clobber=True)
    ## end if

    # Loop over multiplicities, optimizing, until minimum mult is reached
    for mult in range(max_mult, -(max_mult % 2), -2):
        # Drop to next multiplicity if not to be run
        if startvals != None:
            if mult > startvals[0]:
                continue
            ## end if
        ## end if

        # Create/retrieve multiplicity subgroup in repo
        mgp = diagp.require_group(h5_names.mult_prfx + str(mult))

        # Loop over reference multiplicities, from max_mult to current mult
        for ref in range(max_mult, mult-2, -2):
            # Drop to next ref if not to be run
            if startvals != None:
                if ref > startvals[1]:
                    continue
                ## end if
            ## end if

            # Create/retrieve ref multiplicity subgroup
            rgp = mgp.require_group(h5_names.ref_prfx + str(ref))

            # Set & write base string to repo
            base = build_base(m, nm, chg, mult, ref)
            h5_clobber_dataset(rgp, name=h5_names.run_base, data=base, \
                                                            log_clobber=True)

            # If ref is the same as mult, then start fresh. Otherwise, insist
            #  on starting from a prior wavefunction.
            if ref != mult:
                # Build stuff from prior. Start by defining the MOREAD string,
                #  which is unambiguous
                moread_str = build_moread(m, nm, chg, ref, ref)

                # Only worry about the prior geom if basing sep on it
                if fixed_dia_sep:
                    # Just use the fixed value
                    xyz_str = def_dia_xyz(m, nm, init_dia_sep)
                else:
                    # Load the xyz info. Presume two atoms.
                    x = XYZ(path=(build_base(m, nm, chg, ref, ref) + ".xyz"))

                    # Check that sep not too large
                    if x.Dist_single(0,0,1) * PHYS.Ang_per_Bohr > \
                                                            ditch_sep_thresh:
                        # Too large; use default
                        xyz_str = def_dia_xyz(m, nm, init_dia_sep)
                    else:
                        # OK. Use scaled value from prior, if not too small
                        xyz_str = def_dia_xyz(m, nm, \
                        dist=max( \
                            geom_scale * x.Dist_single(0,0,1) * \
                            PHYS.Ang_per_Bohr, init_dia_sep
                                ) )
                    ## end if
                ## end if
            else:
                # Build from scratch if ref == mult
                moread_str = "! NOAUTOSTART"
                xyz_str = def_dia_xyz(m, nm, init_dia_sep)
            ## end if

            # Try executing, looping across convergers. Should probably use a
            #  very small number for %geom MaxIter and perhaps consider a
            #  Calc_Hess true, to try to avoid unbound cases taking forever
            #  to run.
            for conv in convergers:
                # Pause if pause-file touched
                while os.path.isfile(os.path.join(os.getcwd(),pausefname)):
                    sleep(pausetime)
                ## loop

                # Update the reporting name for the converger
                conv_name = (conv if conv != "" else "default") + " & SLOWCONV"

                # Try all convergers with SLOWCONV, clearing temp
                #  files first
                clear_tmp(base)
                oo = exor(template_str, os.getcwd(), [exec_cmd, base], \
                        sim_name=base, \
                        subs=[('OPT', str(opt_str)), \
                            ('CONV', conv + "\n! SLOWCONV"), \
                            ('MULT', str(mult)), \
                            ('CHARGE', str(chg)), \
                            ('MOREAD', moread_str), \
                            ('XYZ', xyz_str)
                            ])[0]
                            ## [0] here b/c execute returns a tuple of objects

                # Check if completed, converged and optimized; store 'last_'
                #  variables and break if so
                if oo.completed and oo.converged and oo.optimized:
                    logger.info(base + " opt converged from '" + \
                                ("model" if ref == mult else \
                                        build_base(m, nm, chg, ref)) + \
                                "' using " + conv_name)
                    h5_clobber_dataset(rgp, name=h5_names.converger, \
                                        data=conv_name, log_clobber=True)
                    break  ## for conv in convergers
                else:
                    logger.warning(base + " opt did not converge using " + \
                                    conv_name)
                ## end if

            else:  ## on for conv in convergers:
                # If never broken, then the whole series of computations failed.
                #  Try one last go with KDIIS and NUMFREQ, clearing any
                #  temp files first, but only if ANFREQ is found in the
                #  template
                if template_str.upper().find('ANFREQ') > -1:
                    clear_tmp(base)
                    oo = exor(template_str.replace("ANFREQ", \
                                    "NUMFREQ \n%freq  Increment 0.01  end"), \
                            os.getcwd(), [exec_cmd, base], \
                            sim_name=base, \
                            subs=[('OPT', str(opt)), \
                                ('CONV', "! KDIIS \n! SLOWCONV "), \
                                ('MULT', str(mult)), \
                                ('CHARGE', str(chg)), \
                                ('MOREAD', moread_str), \
                                ('XYZ', xyz_str)
                                ])[0]

                    # Check if completed, converged and optimized; store 'last_'
                    #  variables if so; if not, log and skip to next
                    #  multiplicity.
                    if oo.completed and oo.converged and oo.optimized:
                        logger.info(base + " opt converged from '" + \
                                ("model" if mult == ref else \
                                        build_base(m, nm, chg, ref)) + \
                                "' using KDIIS & SLOWCONV & NUMFREQ")
                        h5_clobber_dataset(rgp, name=h5_names.converger, \
                                    data="KDIIS & SLOWCONV & NUMFREQ", \
                                    log_clobber=True)
                    else:
                        # Log failure and return.
                        logger.error(base + \
                                " opt failed to converge in all computations.")
                        h5_clobber_dataset(rgp, name=h5_names.converger, \
                                            data="FAILED", log_clobber=True)

                        # Skip remainder of processing of this diatomic
                        return
                    ## end if run succeeded
                else:
                    #  Log failure and return.
                    logger.error(base + \
                                " opt failed to converge in all computations.")
                    h5_clobber_dataset(rgp, name=h5_names.converger, \
                                            data="FAILED", log_clobber=True)

                    # Skip remainder of processing of this diatomic
                    return
                ## end if ANFREQ present in template
            ## next conv

            # Store useful outputs to repo - bond length, final energy, HESS
            #  stuff, dipole moment
            store_dia_results(rgp, oo, XYZ(path=(base + ".xyz")), \
                                                        log_clobber=True)

        ## next ref

    ## next mult

    # Identify the minimum-energy multiplicity and associated properties.
    #  May be inaccurate if key runs fail. parse_dia_mults created to allow re-
    #  running after the fact, once such key runs have been tweaked manually
    #  to re-run satisfactorily.
    parse_dia_mults(diagp, log_clobber=True)

## end def run_dia


def store_dia_results(rgp, oo, xyz, log_clobber=False):
    """ #DOC: store_dia_results docstring
    """

    # Imports
    from opan.const import PHYS
    #TODO: Expand data stored in store_run_results as ORCA_OUTPUT expanded

    #TODO: Implement check(s) for undesirable data here, passing a flag to
    #  calling function to indicate need to halt the particular diatomic.

    # Store the data, overwriting if it exists
    storage_data = (
                (h5_names.out_en, oo.en_last()[oo.EN.SCFFINAL]),
                (h5_names.out_zpe, oo.thermo[oo.THERMO.E_ZPE]),
                (h5_names.out_enth, oo.thermo[oo.THERMO.H_IG]),
                (h5_names.out_dipmom, oo.dipmoms[-1]),
                (h5_names.out_bondlen, \
                                PHYS.Ang_per_Bohr * xyz.Dist_single(0,0,1))
                        )
    for item in storage_data:
        h5_clobber_dataset(rgp, name=item[0], data=item[1], \
                                                    log_clobber=log_clobber)
    ## next item

    # Flush repo
    rgp.file.flush()

## end def store_dia_results


def parse_dia_mults(diagp, log_clobber=False):
    """ #DOC: parse_dia_mults docstring
    """

    # Imports
    from opan.const import atomSym

    # Initialize the minimum energy to zero, which everything should be less
    #  than!  Also init min_mult to ~error value.
    min_en = 0.0
    min_mult = 0
    min_ref = 0

    # Identify lowest-energy multiplicity and wavefunction reference; push
    #  info to repo
    for mg in diagp.values():
        # Check if a given value is a group and that its name matches the
        #  pattern expected of a multiplicity group
        if isinstance(mg, h5.Group) and p_multgrp.search(mg.name) != None:
            # Loop over items in the multiplicity group
            for rg in mg.values():
                # Check if group and name matches ref group pattern
                if isinstance(rg, h5.Group) and \
                                        p_refgrp.search(rg.name) != None:
                    # Check if output energy value exists
                    if not rg.get(h5_names.out_en) == None:
                        # Check if the reported SCF final energy is less than
                        #  the current minimum.
                        if rg.get(h5_names.out_en).value < min_en:
                            # If so, update the minimum and store the
                            #  multiplicity/reference wfn
                            min_en = np.float_(rg.get(h5_names.out_en).value)
                            min_mult = np.int_(p_multgrp.search(mg.name) \
                                                                .group("mult"))
                            min_ref = np.int_(p_refgrp.search(rg.name) \
                                                                .group("ref"))
                        ## end if
                    ## end if
                ## end if
            ## end if
        ## end if
    ## next g

    # Retrieve the appropriate min groups (should never fail, unless NO DATA
    #  whatsoever exists for the diatomic)
    mg = diagp.get(h5_names.mult_prfx + str(min_mult))
    rg = mg.get(h5_names.ref_prfx + str(min_ref))

    # Store the minimum energy and multiplicity, and retrieve/store the other
    #  thermo values
    storage_data = (
                (h5_names.min_en, min_en),
                (h5_names.min_en_mult, min_mult),
                (h5_names.min_en_ref, min_ref),
                (h5_names.min_en_zpe, rg.get(h5_names.out_zpe).value),
                (h5_names.min_en_enth, rg.get(h5_names.out_enth).value),
                (h5_names.min_en_bondlen, rg.get(h5_names.out_bondlen).value),
                (h5_names.min_en_dipmom, rg.get(h5_names.out_dipmom).value)
                    )
    for item in storage_data:
        h5_clobber_dataset(diagp, name=item[0], data=item[1], \
                                                    log_clobber=log_clobber)
    ## next item

    # Flush the repo
    diagp.file.flush()

## end def parse_dia_mults


def merge_h5(f1, f2):
    """ #DOC: merge_h5 docstring
    """

    # Imports

    # Load the files
    hf1 = h5.File(f1)
    hf2 = h5.File(f2)

    # Loop over all root objects in h2, low-level copying to h1. Must check
    #  if each exists in h1, and if so, delete.
    # Start by popping any values out of hf1 if they exist
    [hf1.pop(v.name, None) for v in hf2.values()]

    # Flush hf1 to ensure the disk version is up to date
    hf1.flush()

    # Low-level copy of all hf2 items to hf1
    [h5.h5o.copy(hf2.id, v.name, hf1.id, v.name) for v in hf2.values()]

    # Close everything
    hf2.close()
    hf1.close()

## end def merge_h5


def clear_tmp(base):
    """ #DOC: clear_tmp docstring
    """

    # Imports
    import os

    [os.remove(os.path.join(os.getcwd(), fn)) for fn \
                in os.listdir(os.getcwd()) \
                if (fn.find(base) == 0 and \
                    (fn[-3:] == "tmp" or fn[-5:] == "tmp.0" or \
                        fn[-5:] == "cpout"))]

## end def clear_tmp


def def_dia_xyz(m, nm, dist=init_dia_sep):
    """ #DOC: def_dia_xyz docstring
    """

    # Imports
    from opan.const import atomSym

    # Construct and return the xyz info
    xyz_str = "  " + atomSym[m].capitalize() + "  0 0 0 \n" + \
              "  " + atomSym[nm].capitalize() + "  0 0 " + str(dist)
    return xyz_str


## end def def_dia_xyz


def build_moread(m, nm, chg, mult, ref=None):
    """ #DOC: build_moread docstring
    """

    mo_str = '"' + build_base(m, nm, chg, mult, ref) + '.gbw"'

    mo_str = '! MOREAD\n%moinp ' + mo_str

    return mo_str

## end def build_moread


def build_base(m, nm, chg, mult=None, ref=None):
    """ #DOC: build_base docstring
    """

    # Imports
    from opan.const import atomSym

    base_str = atomSym[m].capitalize() + atomSym[nm].capitalize() + sep + \
                h5_names.chg_prfx + ("_" if chg < 0 else "") + str(abs(chg)) + \
                (h5_names.mult_prfx + str(mult) if mult != None else "") + \
                (h5_names.ref_prfx + str(ref) if ref != None else "")

    return base_str

## end def build_base


def safe_retr(group, dataname, errdesc, log_absent=True):
    """ #DOC: docstring for safe_retr
    """

    # Imports
    import logging

    try:
        workvar = group.get(dataname).value
    except AttributeError:
        if log_absent:
            logging.getLogger(log_names.loggername) \
                    .error(errdesc + " absent from repository.")
        ## end if
        workvar = NA_str
    ## end try

    return workvar

## end def safe_retr


def proof_template(template_str):

    for tag in req_tag_strs:
        if template_str.find(tag) == -1:
            raise(ValueError("'" + tag + "' tag absent from template."))
        ## end if
    ## next tag

## end def proof_template


def setup_logger():
    """ #DOC: setup_logger docstring
    """

    # Imports
    import logging, logging.config

    # Define the configuration dictionary
    dictLogConfig = {
        "version": 1,
        "loggers": {
            log_names.loggername: {
                "handlers": [log_names.handlername],
                "level": "INFO"
                }
            },
        "handlers": {
            log_names.handlername: {
                "class": "logging.FileHandler",
                "formatter": log_names.formattername,
                "filename": os.path.join(os.getcwd(),log_names.filename)
                }
            },
        "formatters": {
            log_names.formattername: {
                "format": log_names.fmt,
                "datefmt": log_names.datefmt
                }
            }
        }

    # Apply the config
    logging.config.dictConfig(dictLogConfig)

## end def setup_logger


def continue_tuple(tupstr):
    """ #DOC: continue_tuple docstring
    """

    # Imports
    import argparse
    from opan.const import atomSym

    # Ensure enclosing parentheses
    workstr = ('(' if tupstr[0] != '(' else "") + \
                tupstr + \
                (')' if tupstr[-1] != ')' else "")

    # Try the tuple strip-and-split conversion
    try:
        out_tup = tuple(map(int,workstr[1:-1].split(',')))
    except:
        msg = '"' + tupstr + '" is not a tuple of integers'
        raise(argparse.ArgumentTypeError(msg))
    ## end try

    # Must be length five
    if not len(out_tup) == 5:
        msg = 'Argument must contain five integers: "' + tupstr + '"'
        raise(argparse.ArgumentTypeError(msg))
    ## end if

    # Must be an implemented metal
    if not out_tup[0] in metals:
        msg = "Metal '" + str(out_tup[0]) + "' (" + atomSym[out_tup[0]] + \
                            ") not implemented"
        raise(argparse.ArgumentTypeError(msg))
    ## end if

    # Must be an implemented nonmetal
    if not out_tup[1] in nonmetals:
        msg = "Nonmetal '" + str(out_tup[1]) + "' (" + atomSym[out_tup[1]] + \
                            ") not implemented"
        raise(argparse.ArgumentTypeError(msg))
    ## end if

    # Return the tuple
    return out_tup

## end def continue_tuple


def h5_clobber_dataset(grp, name, shape=None, dtype=None, \
                                    data=None, log_clobber=False, **kwargs):
    """ #DOC: h5_clobber_dataset docstring.
    """

    # Imports
    import logging, h5py as h5

    # Retrieve logger
    logger = logging.getLogger(log_names.loggername)

    # Check existence of desired object
    if name in grp.keys():
        # Complain if it's not a dataset - WILL NOT FIND A SAME-NAMED GROUP
        if not isinstance(grp.get(name), h5.Dataset):
            # Catches links &c.
            raise(ValueError("Item '" + name + "' is not a dataset"))
        ## end if

        # Clobber and flush repo; log if indicated
        grp.pop(name)
        if log_clobber:
            logger.info("Dataset '" + name + "' in group '" + grp.name + \
                                                        "' was overwritten.")
        ## end if
    ## end if

    # Either way, add new
    grp.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)

## end def h5_clobber_dataset


if __name__ == '__main__':

    # Imports
    import argparse as ap

    # Param name strings
    TMPLT_FILE = 'template_file'
    WKDIR = 'wkdir'
    EXEC = 'exec'
    OPT = 'opt'
    METALS = 'metals'
    NONMETALS = 'nonmetals'
    CATION_NMS = 'cation_nms'
    INIT_SEP_DIA = 'init_sep_dia'
    SKIP_ATOMS = 'skip_atoms'
    CONT_DIA = 'continue_dia'

    # Create the parser
    prs = ap.ArgumentParser(description="Perform Jensen diatomics " + \
                "calculation series.\n\nBackground execution with " + \
                "stderr redirected to a file or to /dev/null " + \
                "is recommended.")

    # Template file argument
    prs.add_argument(TMPLT_FILE, help="Name of input template file")

    # Continuing an errored-out diatomic
    prs.add_argument("--" + CONT_DIA, help="Tuple of information to resume " + \
                                    "computation of a halted diatomic.\n" + \
                                    "Format: (m, nm, chg, mult, ref), " + \
                                    "where 'mult' and 'ref' are the " + \
                                    "values at which to restart computation.",
                                metavar="TUPLE", type=continue_tuple, \
                                default=None)

    # Working directory argument
    prs.add_argument("--" + WKDIR, help="Path to working directory with " + \
            "template file (defaults to current directory)", \
            default=os.getcwd())

    # Execution script
    prs.add_argument("--" + EXEC, help="ORCA execution command (default: " + \
                    exec_cmd + ")", default=None)

    # OPT command
    prs.add_argument("--" + OPT, help="Optimization settings for diatomics " + \
                "(default: '" + opt_str + "')", default=None)

    # Overwrites of metals and nonmetals (incl cation ones)
    prs.add_argument("--" + METALS, help="Metals to run; pass as " + \
                "stringified array of atomic numbers (e.g., '[21, 23, 29]';" + \
                " default: " + str(sorted(list(metals))) + ". Only Sc-Zn " + \
                "currently supported.", default=None)
    prs.add_argument("--" + NONMETALS, help="Nonmetals to run; pass as " + \
                "stringified array of atomic numbers (e.g., '[1, 6, 9]'; " + \
                "default: " + str(sorted(list(nonmetals))) + ". Only these " + \
                "defaults currently supported.", default=None)
    prs.add_argument("--" + CATION_NMS, help="Nonmetals to run as " + \
                "monopositive cations; pass as " + \
                "stringified array of atomic numbers (e.g., '[1, 8]'; " + \
                "this is the default). Any nonmetals are valid.", default=None)

    # Twiddlers for execution settings
    prs.add_argument("--" + INIT_SEP_DIA, help="Initial separation between " + \
                "atoms in diatomic computations, in Angstroms. " + \
                "Default is " + str(init_dia_sep) + " Angstroms.", default=None)

    prs.add_argument("--" + SKIP_ATOMS, help="Skip atom computations, " + \
                "generally for debugging purposes.", default=False, \
                const=True, action='store_const')

    # Make the param namespace and retrieve the params dictionary
    ns = prs.parse_args()
    params = vars(ns)

    # Overwrite the metals, nonmetals, cations, initial diatomic separation,
    #  etc. if specified
    if not params[METALS] == None:
        metals = set(eval(params[METALS]))
    ## end if
    if not params[NONMETALS] == None:
        nonmetals = set(eval(params[NONMETALS]))
    ## end if
    if not params[CATION_NMS] == None:
        cation_nms = set(eval(params[CATION_NMS]))
    ## end if
    if not params[INIT_SEP_DIA] == None:
        init_dia_sep = float(params[INIT_SEP_DIA])
    ## end if
    if not params[OPT] == None:
        opt_str = params[OPT]
    ## end if
    if not params[EXEC] == None:
        exec_cmd = params[EXEC]
    ## end if
    skip_atoms = bool(params[SKIP_ATOMS])

    # Check for whether a diatomic restart was requested
    if params[CONT_DIA] == None:
        # Execute the run
        do_run(params[TMPLT_FILE], wkdir=params[WKDIR])
    else:
        # Execute a diatomic restart
        continue_dia(params[TMPLT_FILE], *params[CONT_DIA], \
                        wkdir=params[WKDIR])
    ## end if

## end if main


