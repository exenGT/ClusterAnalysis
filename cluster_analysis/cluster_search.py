from collections import deque
import numpy as np


def path_search(main_neigh_aux_indices_dict,
                aux_neigh_main_indices_dict,
                num_unit,
                num_at_unit, 
                unit_main_atoms_indices,
                unit_aux_atoms_indices):

    """
    Find all paths (clusters) in the system, using Breadth First Search (BFS) algorithm.

    Args:

    main_neigh_aux_indices_dict (dict):
    dictionary of neighboring aux ion indices for every main ion

    aux_neigh_main_indices_dict (dict): 
    dictionary of neighboring main ion indices for every aux ion

    num_unit (int):  number of salt ions

    num_at_unit (int):  number of atoms in a salt ion

    unit_main_atoms_indices (list int):
    list of atom indices in a single main ion

    unit_aux_atoms_indices (list int):
    list of atom indices in a single aux ion

    Return:

    mro_atom_indices (numpy.ndarray): 
    array of all atom indices within a cluster of length >= 2

    cluster_sizes (dict): 
    histogram of the number of clusters with different lengths

    N_main_N_aux (list):
    list of populations of main and aux ions

    """

    ## initialize mro atom indices and cluster lengths
    mro_atom_indices = []
    cluster_sizes = []

    ## initialize main atom indices and aux ion indices
    main_ion_indices = list(range(num_unit))
    aux_ion_indices = list(range(num_unit))

    ## initialize cluster length for each main and aux ion index
    main_ion_indices_cluster_sizes = np.zeros(num_unit)
    aux_ion_indices_cluster_sizes = np.zeros(num_unit)

    ## initialize (number of main ions, number of aux ions) in each cluster
    N_main_N_aux = []

    ## initialize the queue for main ions
    path_queue_main = deque()

    ## initialize the unvisited main ions and aux ions
    main_ion_indices_unvisited = main_ion_indices.copy()
    aux_ion_indices_unvisited = aux_ion_indices.copy()

    ## initialize the visit status of main atoms and aux ions
    main_ion_indices_is_visited = np.zeros(len(main_ion_indices_unvisited), dtype=bool)
    aux_ion_indices_is_visited = np.zeros(len(aux_ion_indices_unvisited), dtype=bool)


    ## start the BFS algorithm

    ## while there are still unvisited main atoms
    while main_ion_indices_unvisited:

        ## re-initialize cluster atom indices
        cluster_atom_indices = []

        ## re-initialize visited atom indices
        main_ion_indices_visited = []
        aux_ion_indices_visited = []

        ## select the root main ion
        start_main_ion_index = main_ion_indices_unvisited[0]

        ## put the root main ion into the main queue
        path_queue_main.append(start_main_ion_index)

        ## re-initialize cluster size
        cluster_size = 0

        ## re-initialize population of main ions and aux ions
        ## in the cluster
        N_main = 0
        N_aux = 0


        ## while the queue is not empty:
        while path_queue_main:

            ## get a main ion first-in-first-out (FIFO) from the queue
            main_ion_index = path_queue_main.popleft()

            ## record this main ion as visited
            main_ion_indices_visited.append(main_ion_index)
            main_ion_indices_unvisited.remove(main_ion_index)

            ## increase cluster size by one
            cluster_size += 1
            ## increase number of main ions in this cluster by one
            N_main += 1

            ## find neighboring aux ions of this main ion
            main_neigh_aux_indices = main_neigh_aux_indices_dict[main_ion_index]

            ## enter next round of search if this main atom has no neighbor aux
            if main_neigh_aux_indices.size == 0:
                break

            ## for each neighboring aux ion of this main atom
            for aux_ion_index in main_neigh_aux_indices:

                aux_ion_indices_cluster_sizes[aux_ion_index] = cluster_size

                ## if this aux ion is not visited:
                if not aux_ion_indices_is_visited[aux_ion_index]:

                    ## add this aux ion to the visited aux ions
                    aux_ion_indices_visited.append(aux_ion_index)
                    aux_ion_indices_unvisited.remove(aux_ion_index)
                    aux_ion_indices_is_visited[aux_ion_index] = True

                    ## increase number of aux ions in this cluster by one
                    N_aux += 1

                ## find neighboring main ions of this aux ion
                aux_neigh_main_indices = aux_neigh_main_indices_dict[aux_ion_index]

                ## consider only those unvisited ones
                aux_neigh_main_indices = np.setdiff1d(aux_neigh_main_indices, np.array(main_ion_indices_visited))

                ## go to next aux ion if this aux ion has no neighboring main ions
                if aux_neigh_main_indices.size == 0:
                    continue

                ## for each neighboring main ion of this aux ion in the cluster
                for main_ion_index_2 in aux_neigh_main_indices:

                    ## if this main ion is not visited
                    if not main_ion_indices_is_visited[main_ion_index_2]:

                        ## add this main ion to the queue
                        path_queue_main.append(main_ion_index_2)

                        ## add this main ion to the visited main ions
                        main_ion_indices_is_visited[main_ion_index_2] = True


        cluster_sizes.append(cluster_size)
        N_main_N_aux.append([N_main, N_aux])

        ## add clusters of size > 1
        for main_ion_index in main_ion_indices_visited:

            cluster_atom_indices += list(main_ion_index*num_at_unit \
                                       + np.array(unit_main_atoms_indices))

        for aux_ion_index in aux_ion_indices_visited:

            cluster_atom_indices += list(aux_ion_index*num_at_unit \
                                       + np.array(unit_aux_atoms_indices))


        mro_atom_indices.append((N_main, N_aux, cluster_atom_indices))


    ## add the zero cluster lengths for dissociated aux ions
    for aux_index in aux_ion_indices_unvisited:

        cluster_sizes.append(0)
        N_main_N_aux.append([0, 1])

        mro_atom_indices.append((0, 1, \
                                 list(aux_index*num_at_unit + np.array(unit_aux_atoms_indices))))

    ## print important information
    print("cluster_sizes = {}".format(cluster_sizes))

    print("cluster size sum = {}".format(np.sum(cluster_sizes)))
    print("cluster size mean = {}".format(np.mean(cluster_sizes)))

    print("number of clusters = {}".format(len(cluster_sizes)))


    return mro_atom_indices, cluster_sizes, N_main_N_aux

