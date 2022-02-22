### Code v1 finished on 11/24/2020 by Jingyang Wang
### debugged on 02/21/2021
### based on neighborlist.py of ASE package

import numpy as np
import sys

######################################################################

def get_neighbors(atoms, rcut, output=''):

    """
    Gets all neighbors within distance rcut from every atom in supercell
    Use cell list + neighbor list

    Args:

    atoms (ase atoms): ASE atoms object
    rcut (float): cutoff distance
    output (string): variables to be output

    Return:

    neigh_at_index (list of list int): list of neighbor atom indices
    neigh_at_dists_dict (dict, optional): dictionary of neighbor atom distances
    neigh_at_vecs_dict (dict, optional): dictionary of neighbor atom vectors
    """

    ## check if keyword "output" has valid values

    if not ((output == '') or (output == 'd') or (output == 'v') or (output == 'dv') or (output == 'vd')):

        print("Invalid output; should be one of the following:\n'', d, v, dv, vd.")
        sys.exit(1)


    ### (1) construct bins

    def build_bins(latt, rcut):

        """
        Builds list of bins for neighbor list search
    
        Args:
    
        latt: cell lattice of the structure
        rcut: cutoff distance for neighbor search
    
        Returns:
    
        num_bins_3D: list of number bins in each lattice direction
        """
        
        recp_latt = latt.reciprocal()
    
        L_c = 1 / np.linalg.norm(recp_latt, axis=1)
    
        num_bins_3D = np.maximum((L_c / rcut).astype(int), [1, 1, 1])
    
        return num_bins_3D


    num_bins_3D = build_bins(latt=atoms.cell, rcut=rcut)

    tot_num_bins = np.prod(num_bins_3D)


    ### (2) place atoms into each bin

    ## get positions and scaled positions of atoms
    Cart_coords = atoms.get_positions(wrap=True)
    frac_coords = atoms.get_scaled_positions(wrap=True)

    ## place each atom into a bin
    ## find the 3D bin index for each atom
    at_bins_3D_index = np.floor(frac_coords * num_bins_3D).astype(int)

    ## find the 1D bin index for each atom
    at_bins_1D_index = at_bins_3D_index[:, 0] + \
                       num_bins_3D[0] * (at_bins_3D_index[:, 1] + \
                                      num_bins_3D[1] * at_bins_3D_index[:, 2])

    ## sort at_bins_1D_index according to bins_1D_index
    at_index_in_sorted_1D_bins = np.argsort(at_bins_1D_index)
    at_bins_1D_index_sorted = at_bins_1D_index[at_index_in_sorted_1D_bins]

    # get max number of atoms per bin
    max_natoms_per_bin = np.bincount(at_bins_1D_index_sorted).max()

    # initialize array of atom indices grouped by 1D bins (with placeholder values = -1)
    at_index_grped_by_1D_bins = -np.ones((tot_num_bins, max_natoms_per_bin), dtype=int)

    # fill this array with atom indices
    for i in range(max_natoms_per_bin):

        # Create a mask array that identifies ** the first atom of each bin **.
        mask = np.append([True], at_bins_1D_index_sorted[:-1] != at_bins_1D_index_sorted[1:])
        
        # Assign all first atoms.
        at_index_grped_by_1D_bins[at_bins_1D_index_sorted[mask], i] = at_index_in_sorted_1D_bins[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        at_index_in_sorted_1D_bins = at_index_in_sorted_1D_bins[mask]
        at_bins_1D_index_sorted = at_bins_1D_index_sorted[mask]


    mask = at_index_grped_by_1D_bins != -1

    num_atoms_grped_by_1D_bins = np.count_nonzero(mask, axis=1)

    at_index_expanded = at_index_grped_by_1D_bins[mask]

    
    ### (3) get neighboring bins

    # neighs contains all 3D products of -1, 0, 1 (27 terms)
    bins_3D_index = np.mgrid[0 : num_bins_3D[0], \
                             0 : num_bins_3D[1], \
                             0 : num_bins_3D[2]].T.reshape(-1,3)

    bins_1D_index = bins_3D_index[:, 0] + \
                    num_bins_3D[0] * (bins_3D_index[:, 1] + \
                                      num_bins_3D[1] * bins_3D_index[:, 2])

    neighs_3D_index = np.mgrid[-1:2, -1:2, -1:2].T.reshape(-1,3)

    nc_bins_3D_index = bins_3D_index[:, None, :] + neighs_3D_index[None, :, :]


    ### (4) check periodicity:
    ###     if periodic in a direction: divmod
    ###     if not periodic in a direction: clip

    pbc = atoms.get_pbc()

    cell_shift_3D = np.zeros_like(nc_bins_3D_index)
    nc_bins_3D_index_wrapped = np.zeros_like(nc_bins_3D_index)
    nc_bins_3D_index_nopbc_wrapped = np.copy(nc_bins_3D_index)

    for dim in range(3):

        if pbc[dim]:

            # Direction "dim" is periodic.
            cell_shift_3D[:, :, dim], nc_bins_3D_index_wrapped[:, :, dim] = \
                divmod(nc_bins_3D_index[:, :, dim], num_bins_3D[dim])

        else:

            # Direction "dim" is non-periodic.
            nc_bins_3D_index_wrapped[:, :, dim] = \
                np.clip(nc_bins_3D_index[:, :, dim], 0, num_bins_3D[dim]-1)

            nc_bins_3D_index_nopbc_wrapped[:, :, dim] = nc_bins_3D_index_wrapped[:, :, dim]


    nc_bins_1D_index_wrapped = nc_bins_3D_index_wrapped[:, :, 0] + \
                               num_bins_3D[0] * (nc_bins_3D_index_wrapped[:, :, 1] + \
                                                 num_bins_3D[1] * nc_bins_3D_index_wrapped[:, :, 2])


    ### (5) Construct list of atoms in neighboring cells, and shifts of neighboring cells

    at_index_in_nc = at_index_grped_by_1D_bins[nc_bins_1D_index_wrapped]

    ### set up a mask which indicates the existence of an atom (!= -1)
    mask = at_index_in_nc != -1

    ### number of "True" in mask
    num_atoms_in_nc = np.count_nonzero(mask, axis=2)

    for ic in range(num_atoms_in_nc.shape[0]):
    
        unique_shifts_inc_per_ic = np.unique(nc_bins_3D_index_nopbc_wrapped[ic], axis=0, return_index=True)[1] # CORRECTED CODE
        repeated_shifts_inc_per_ic = list(set(range(27)) - set(unique_shifts_inc_per_ic))
    
        ## remove those repeated atoms
        num_atoms_in_nc[ic][repeated_shifts_inc_per_ic] = 0
        at_index_in_nc[ic][repeated_shifts_inc_per_ic] = -1
    
    ### reset the mask for repeated atoms due to non-periodicity
    mask = at_index_in_nc != -1

    at_index_in_nc_expanded = at_index_in_nc[mask]

    tot_num_atoms_in_nc_per_bin = np.sum(num_atoms_in_nc, axis=1)

    #####################################################################################

    ### https://stackoverflow.com/questions/64955019/return-a-numpy-array-with-numbers-of-elements-specified-in-another-array

    cell_shift_3D = cell_shift_3D.reshape(-1, 3)
    num_atoms_in_nc = num_atoms_in_nc.reshape(-1)

    at_cell_shift_3D_expanded = np.repeat(cell_shift_3D, num_atoms_in_nc, axis=0)

    at_cell_shift_3D_set = {tuple(shift) for shift in at_cell_shift_3D_expanded}

    ## initialize dictionaries
    neigh_at_vecs_dict = dict()
    neigh_at_dists_dict = dict()


    N_at = len(atoms)

    for shift in at_cell_shift_3D_set:

        if 'v' in output:
            neigh_at_vecs_dict[shift] = np.zeros((N_at, N_at, 3))
            
        if 'd' in output:
            neigh_at_dists_dict[shift] = np.zeros((N_at, N_at))

    #####################################################################################

    ### (6) calculate vectors and distances between neighbor-cell atoms

    cumu_num_atoms_grped_by_1D_bins = np.cumsum(num_atoms_grped_by_1D_bins)
    cumu_tot_num_atoms_in_nc_per_bin = np.cumsum(tot_num_atoms_in_nc_per_bin)


    def pairwise_vecs_dists(Cart_coords_1, Cart_coords_2, shift_2, latt):

        """
        Function that finds the pairwise distances and vectors between two coordinate lists,
        considering the periodic shift

        Args:

        Cart_coords_1: list 1
        Cart_coords_2: list 2
        shift_2: shift 2
        latt: cell lattice

        Returns:

        vecs_12:  array of pairwise vectors between Cart_coords_1 and Cart_coords_2
        dists_12: array of pairwise distances between Cart_coords_1 and Cart_coords_2

        """

        shifted_Cart_coords_2 = Cart_coords_2 + shift_2.dot(latt)

        vecs_12 = shifted_Cart_coords_2[None, :, :] - Cart_coords_1[:, None, :]

        dists_12 = np.linalg.norm(vecs_12, axis=-1)

        return vecs_12, dists_12


    ### prepare list of neighboring atom indices for each atom
    neigh_at_index = [[] for _ in range(N_at)]

    cumu_num_atoms_at_1_prev = 0
    cumu_num_atoms_at_2_prev = 0


    for ic in range(len(num_atoms_grped_by_1D_bins)):

        cumu_num_atoms_at_1 = cumu_num_atoms_grped_by_1D_bins[ic]
        cumu_num_atoms_at_2 = cumu_tot_num_atoms_in_nc_per_bin[ic]

        index_at_1 = at_index_expanded[cumu_num_atoms_at_1_prev : cumu_num_atoms_at_1]


        if index_at_1.size > 0:

            index_at_2 = at_index_in_nc_expanded[cumu_num_atoms_at_2_prev : cumu_num_atoms_at_2]
        
            Cart_coords_at_1 = Cart_coords[index_at_1]
    
            Cart_coords_at_2 = Cart_coords[index_at_2]
    
            cell_shift_at_2 = at_cell_shift_3D_expanded[cumu_num_atoms_at_2_prev : cumu_num_atoms_at_2, :]
    
            index_12 = np.array(np.meshgrid(index_at_1, index_at_2)).T

            vecs_12, dists_12 = pairwise_vecs_dists(Cart_coords_at_1, Cart_coords_at_2, \
                                                    cell_shift_at_2, atoms.cell[:])

            #### ---------------------------------------------------------

            ## choose atoms with distance less than cutoff
            mask = np.logical_and(dists_12 <= rcut, dists_12 > 0.)

            ## fill in corresponding index, vectors, and distance arrays
            for i_1 in range(index_12.shape[0]):
    
                at_1 = index_12[i_1, 0, 0]
                i_2_mask = mask[i_1]
        
                neigh_at_index[at_1] = index_12[i_1, i_2_mask, 1]

                ##############################################
                ### Note:
                ### neigh_at_vecs should be (N*N*3)
                ### neigh_at_dists should be (N*N)

                for i_2 in np.where(i_2_mask)[0]:

                    at_2 = index_12[i_1, i_2, 1]
                    shift_i_2 = tuple(cell_shift_at_2[i_2])

                    if 'v' in output:
                        neigh_at_vecs_dict[shift_i_2][at_1, at_2, :] = vecs_12[i_1, i_2, :]

                    if 'd' in output:
                        neigh_at_dists_dict[shift_i_2][at_1, at_2] = dists_12[i_1, i_2]

                ##############################################

        ### update indices
        cumu_num_atoms_at_1_prev = cumu_num_atoms_at_1
        cumu_num_atoms_at_2_prev = cumu_num_atoms_at_2


    ## return final results
    if output == '':
        return neigh_at_index

    elif output == 'd':
        return neigh_at_index, neigh_at_dists_dict

    elif output == 'v':
        return neigh_at_index, neigh_at_vecs_dict

    elif (output == 'vd') or (output == 'dv'):
        return neigh_at_index, neigh_at_dists_dict, neigh_at_vecs_dict
    
