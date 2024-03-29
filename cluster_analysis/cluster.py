import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import ase.io.lammpsdata
import ase.io.lammpsrun
from ase.io import write

import argparse
from argparse import RawTextHelpFormatter
import textwrap
from distutils.util import strtobool

from collections import defaultdict

from .neigh_search import get_neighbors
from .cluster_search import path_search


"""
Find the distribution of clusters in the structure
"""

###########################################################

def cluster(image, 
            species_main_marker, 
            species_aux_marker, 
            num_unit, 
            num_at_unit,
            unit_main_atoms_indices,
            unit_aux_atoms_indices):

    """
    Function that returns list of populations of the main and auxiliary ions
    in a salt configuration


    Args:

    image (ase.atoms):  configuration containing salt ions

    species_main_marker (string):  name of the main ion species

    species_aux_marker (string):  name of the auxiliary ion species

    num_unit (int):  number of salt "units" in the configuration

    num_at_unit (int):  number of atoms in a salt unit

    unit_main_atoms_indices (list int): list of atom indices in a single main ion

    unit_aux_atoms_indices (list int): list of atom indices in a single aux ion


    Return:

    cluster_atom_indices (numpy.ndarray):
    array of all atom indices within a cluster of length >= 2

    cluster_sizes (dict):
    histogram of the number of clusters with different lengths

    N_main_N_aux (list):
    list of populations of main and aux ions
    """

    ## get species array
    species = np.array(image.get_chemical_symbols())

    ## generate neighbor list for all atoms
    all_neigh_indices = get_neighbors(image, rcut=2.2)

    ## find all atoms of species "species_main_marker"
    main_marker_indices = np.array([atom.index for atom in image \
        if atom.symbol == species_main_marker]).astype(int)

    ## find all atoms of species "species_aux_marker"
    aux_marker_indices = np.array([atom.index for atom in image \
        if atom.symbol == species_aux_marker]).astype(int)

    ##-----------------------------------------------------------

    def main_aux_neighs(all_neigh_indices, 
                        main_marker_indices, 
                        species, 
                        species_aux_marker, 
                        num_at_unit):

        """
        Function that returns the main-auxiliary ion neighbor dictionary
        
        Args:

        all_neigh_indices (list int): neighbor list for all atoms
        main_marker_indices (numpy.ndarray int): array of indices of main marker atoms
        species (numpy.adarray string): array of species strings for all atoms
        species_aux_marker (string): species string of the auxiliary marker atom
        num_at_unit (int): number of atoms in one salt "unit"

        Return:

        main_neigh_aux_indices_dict (dict): dictionary of neighbor auxiliary marker atom indices
                                            for each main marker atom
        """

        ## initialize dict of neighboring atoms
        main_neigh_aux_indices_dict = defaultdict(list)

        ## for every main marker atom in the system
        for main_marker_index in main_marker_indices:
    
            ## find the indices of its nearest neighbor "aux" marker atoms
            main_neigh_indices = all_neigh_indices[main_marker_index].astype(int)
    
            main_neigh_aux_indices = main_neigh_indices\
                                    [species[main_neigh_indices] == species_aux_marker]
    
            ## remove repeating aux ion indices
            ## (since there may be multiple marker atoms in an "aux" ion)
            main_neigh_aux_indices = np.unique(main_neigh_aux_indices // num_at_unit).astype(int)
    
            ## add this neighbor aux ion list to:
            ## main_neigh_aux_indices_dict[main_ion_index]
            main_ion_index = main_marker_index // num_at_unit

            main_neigh_aux_indices_dict[main_ion_index] = \
                np.union1d(main_neigh_aux_indices_dict[main_ion_index], \
                           main_neigh_aux_indices).astype(int)


        return main_neigh_aux_indices_dict


    main_neigh_aux_indices_dict = main_aux_neighs(all_neigh_indices,
                                                  main_marker_indices,
                                                  species,
                                                  species_aux_marker,
                                                  num_at_unit)

    aux_neigh_main_indices_dict = main_aux_neighs(all_neigh_indices,
                                                  aux_marker_indices,
                                                  species,
                                                  species_main_marker,
                                                  num_at_unit)


    ## find all clusters in the configuration
    clusters_atom_indices, clusters_size, N_main_N_aux \
      = path_search(main_neigh_aux_indices_dict, 
                    aux_neigh_main_indices_dict, 
                    num_unit, 
                    num_at_unit,
                    unit_main_atoms_indices,
                    unit_aux_atoms_indices)

    return clusters_atom_indices, clusters_size, N_main_N_aux


###########################################################################

def analyze(args_list=None):

    """
    Function that analyzes the cluster from the molecular dynamics dump files.

    Args:
    args_list (list): List of arguments (possibly read from command line prompt)
    """

    ## initialize system-specific inputs via command line argument
    parser = argparse.ArgumentParser(description='Main function of cluster analysis.', 
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-p', '--path',
                        type=str,
                        required=False, 
                        metavar='path',
                        help='Working path directory. (Default: current directory)')

    parser.add_argument('-n', '--num_dump',
                        type=int,
                        required=True,
                        metavar='num_dump',
                        help='Number of dump files to be analyzed.')

    parser.add_argument('-d', '--dump_dir',
                        type=str,
                        required=True,
                        metavar='dump_dir',
                        help='Directory of dump files.')

    parser.add_argument('-i', '--info_file',
                        type=str,
                        required=True, 
                        metavar='info_file',
                        help=textwrap.dedent('''\
                              Name of file containing basic information needed. Should be located in path.
                              Format:
                              1st line: number of salt units
                              2nd line: number of atoms per salt unit
                              3rd line: main ion indices
                              4th line: aux ion indices
                              5th line: species string of main marker atom
                              6th line: species string of auxiliary marker atom'''))

    parser.add_argument('-u', '--salt_unit_file',
                        type=str,
                        required=True, 
                        metavar='salt_unit_file',
                        help=textwrap.dedent('''\
                             Name of structure file of the salt unit. Should be located in path.
                             Format: *.xyz'''))

    parser.add_argument('-c', '--output_clusters_coords',
                        # type=lambda x: bool(strtobool(x)),
                        type=bool,
                        default=True,
                        metavar='salt_unit_file',
                        help=textwrap.dedent('''\
                             Whether to output clusters coordinates into *.cif files.'''))

    parser.add_argument('-s', '--output_clusters_size',
                        # type=lambda x: bool(strtobool(x)),
                        type=bool,
                        default=True,
                        metavar='salt_unit_file',
                        help=textwrap.dedent('''\
                             Whether to output clusters size into a text file.'''))

    parser.add_argument('-r', '--output_gyration_radius',
                        # type=lambda x: bool(strtobool(x)),
                        type=bool,
                        default=True,
                        metavar='salt_unit_file',
                        help=textwrap.dedent('''\
                             Whether to output cluster radius of gyration into a text file.'''))

    args = parser.parse_args(args_list)

    if args.path:
        path = args.path
    else:
        path = os.getcwd()

    try:
        os.chdir(path)
    except (FileNotFoundError, NotADirectoryError):
        print("Invalid path!")
        sys.exit(1)

    ## read information from info_file
    info_file = args.info_file

    with open(info_file) as f:

        for i, line in enumerate(f):

            if i == 0:
                num_units = int(line.strip())

            elif i == 1:
                num_at_unit = int(line.strip())

            elif i == 2:
                unit_main_atoms_indices = np.array(line.strip().split(",")).astype(int)

            elif i == 3:
                unit_aux_atoms_indices = np.array(line.strip().split(",")).astype(int)

            elif i == 4:
                species_main_marker = line.strip()

            elif i == 5:
                species_aux_marker = line.strip()


    ## initialize species
    salt_unit_file = args.salt_unit_file
    
    image_salt_unit = ase.io.read(salt_unit_file, format="xyz")
    
    species_salt_unit = np.array(image_salt_unit.get_chemical_symbols())

    species = np.tile(species_salt_unit, num_units)

    ## create a directory for generated cluster structure files
    if not os.path.exists("images_cluster"):
        os.makedirs("images_cluster")

    ## read the configurations
    dump_dir = args.dump_dir

    try:
        os.chdir(args.dump_dir)
    except (FileNotFoundError, NotADirectoryError):
        print("Invalid dump_files path!")
        sys.exit(2)

    num_dump = args.num_dump
    dump_range = range(0, num_dump)

    ## initialize empty size lists
    N_main_N_aux_accumu = [[]]
    R_g_accumu = []

    
    for i, image_num in enumerate(dump_range):
    
        print("\nimage # {}:\n".format(i))
    
        image_index = str(image_num)
    
        with open(image_index+".dump", "r") as f:
        
            image = ase.io.lammpsrun.read_lammps_dump_text(f)
    
        ## process image in ase atoms class
        image.set_chemical_symbols(symbols=species)
    
        image.set_pbc(True)
        image.wrap()

        Lx, Ly, Lz = image.get_cell_lengths_and_angles()[0:3]
    
        clusters_atom_indices, \
        clusters_size, \
        N_main_N_aux = cluster(image=image,
                               species_main_marker=species_main_marker,
                               species_aux_marker=species_aux_marker,
                               num_unit=num_units,
                               num_at_unit=num_at_unit,
                               unit_main_atoms_indices=unit_main_atoms_indices,
                               unit_aux_atoms_indices=unit_aux_atoms_indices)

        ### write each cluster to cif file

        if len(clusters_atom_indices) > 0:
        
            os.chdir("../images_cluster")

            R_g = []

        
            for i_cluster, cluster_info in enumerate(clusters_atom_indices):
        
                N_main, N_aux, atom_indices = cluster_info
            
                ## build a new atoms object from the atoms with indices "neigh_atom_inds"
                image_cluster = image[atom_indices]

                ## unwraps the coordinates across the periodic boundaries
                coords = image_cluster.get_positions()
                # print("coords.shape = {}".format(coords.shape))

                if coords.shape[0] > 1:

                    X, Y, Z = coords.T

                    X_sorted_inds = np.argsort(X)
                    Y_sorted_inds = np.argsort(Y)
                    Z_sorted_inds = np.argsort(Z)

                    X_sorted = X[X_sorted_inds]
                    Y_sorted = Y[Y_sorted_inds]
                    Z_sorted = Z[Z_sorted_inds]

                    delta_X = X_sorted[1:] - X_sorted[:-1]
                    delta_Y = Y_sorted[1:] - Y_sorted[:-1]
                    delta_Z = Z_sorted[1:] - Z_sorted[:-1]

                    max_sep = 10.

                    X_gap = np.argwhere(delta_X > max_sep)
                    Y_gap = np.argwhere(delta_Y > max_sep)
                    Z_gap = np.argwhere(delta_Z > max_sep)


                    if X_gap:

                        X_bound_gt_ind = X_gap[0, 0] + 1

                        X_le_inds = X_sorted_inds[:X_bound_gt_ind]
                        X_gt_inds = X_sorted_inds[X_bound_gt_ind:]

                        # print("X = {}".format(X))

                        X[X_le_inds] += Lx / 2.
                        X[X_gt_inds] -= Lx / 2.

                        # print("X = {}".format(X))


                    if Y_gap:

                        Y_bound_gt_ind = Y_gap[0, 0] + 1

                        Y_le_inds = Y_sorted_inds[:Y_bound_gt_ind]
                        Y_gt_inds = Y_sorted_inds[Y_bound_gt_ind:]

                        # print("Y = {}".format(Y))

                        Y[Y_le_inds] += Ly / 2.
                        Y[Y_gt_inds] -= Ly / 2.

                        # print("Y = {}".format(Y))


                    if Z_gap:

                        Z_bound_gt_ind = Z_gap[0, 0] + 1

                        Z_le_inds = Z_sorted_inds[:Z_bound_gt_ind]
                        Z_gt_inds = Z_sorted_inds[Z_bound_gt_ind:]

                        # print("Z = {}".format(Z))

                        Z[Z_le_inds] += Lz / 2.
                        Z[Z_gt_inds] -= Lz / 2.

                        # print("Z = {}".format(Z))

    
                    new_coords = np.vstack((X, Y, Z)).T
    
                    image_cluster.set_positions(new_coords)
    
                    # print("--------------------------------")

                ### DEBUG ###

                ## calculate the radius of gyration

                com = image_cluster.get_center_of_mass()
                centered_pos = image_cluster.get_positions() - com
                centered_pos_dot = np.sum(centered_pos*centered_pos, axis=1)
                r_g = np.sqrt(np.mean(centered_pos_dot))
                R_g.append(r_g)
                # print("com = {}".format(com))
                # print("centered_pos = {}".format(centered_pos))
                # print("centered_pos_dot = {}".format(centered_pos_dot))
                # print("r_g = {}".format(r_g))


                ### DEBUG ###
        
                ## only write clusters in the first image
                if (i == 0):
                    write('image_' + str(i) + '_cluster_' + str(i_cluster) + \
                          '_' + str(N_main) + '-' + str(N_aux) + '.cif', image_cluster)
        
            os.chdir("../" + dump_dir)


        if args.output_clusters_size:

            try:
                N_main_N_aux_accumu = np.concatenate((N_main_N_aux_accumu, N_main_N_aux), axis=0)
        
            except ValueError:
                N_main_N_aux_accumu = N_main_N_aux


        if args.output_gyration_radius:

            print("R_g = {}".format(R_g))

            try:
                R_g_accumu = np.concatenate((R_g_accumu, R_g), axis=0)

            except ValueError:
                R_g_accumu = R_g


    ## save the cluster distribution as text file
    os.chdir("..")

    if args.output_clusters_size:
        np.savetxt("clusters_size_hist.txt", N_main_N_aux_accumu, fmt='%d,%d')

    if args.output_gyration_radius:
        np.savetxt("clusters_rg_hist.txt", R_g_accumu, fmt='%.2f')

