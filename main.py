from cluster_analysis.cluster import analyze
from cluster_analysis.histplot import plot_hist

# analyze()

# analyze(args_list=['--path', "/Users/jw598/Cui_group_research/Machine_learning_CE/codes/analysis/EL/ClusterAnalysis/cluster_analysis/data/real",
#                    '--num_dump', "5",
#                    '--dump_dir', "dump_files",
#                    '--info_file', "info_file.txt",
#                    '--salt_unit_file', "LiFSI.xyz"])

analyze(args_list=['--path', "/Users/jw598/Cui_group_research/Machine_learning_CE/jobs/solutions/EL5_large/300K_analysis",
                   '--num_dump', "5",
                   '--dump_dir', "dump_salt",
                   '--info_file', "info_file.txt",
                   '--salt_unit_file', "LiFSI.xyz"])

# plot_hist(path="/Users/jw598/Cui_group_research/Machine_learning_CE/jobs/solutions/EL5_large/300K_analysis/dump_salt",
#           hist_file="cluster_sizes_hist.txt", 
#           main_str="Li$^{+}$", 
#           aux_str="FSI$^{-}$", 
#           fig_name="EL5_300K_hist",
#           fig_height=7.0,
#           fig_width=6.0,
#           colorbar=True,
#           text=False,
#           n_max=16,
#           v_min=0.,
#           v_max=0.2)

# analyze(args_list=['--path', "/Users/jw598/Cui_group_research/Xin_Gao_project_2/jobs/solutions/FDMB",
#                    '--num_dump', "100",
#                    '--dump_dir', "dump_salt",
#                    '--info_file', "info_file.txt",
#                    '--salt_unit_file', "Li2S6.xyz"])

# for syst in ["DOL-DME", "DEE", "F3DEE", "F4DEE", "F5DEE", "F6DEE", "FDMB"]:

#     plot_hist(path="/Users/jw598/Cui_group_research/Xin_Gao_project_2/jobs/solutions/"+syst,
#               hist_file="cluster_sizes_hist.txt", 
#               main_str="Li$^{+}$", 
#               aux_str="S$_6^{2-}$", 
#               fig_name=syst+"_hist",
#               n_max=5)
