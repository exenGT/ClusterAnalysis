import unittest
import os

class TestAnalysisMethods(unittest.TestCase):

    def test_run(self):

        from cluster_analysis.cluster import analyze

        analyze(args_list=['--path', "/Users/jw598/Cui_group_research/Machine_learning_CE/codes/analysis/EL/ClusterAnalysis/cluster_analysis/data/test",
                   '--num_dump', "1",
                   '--dump_dir', "dump_file_test",
                   '--info_file', "info_file.txt",
                   '--salt_unit_file', "LiFSI.xyz"])
        

    def test_hist(self):
    	"""
        Test function for histogram text file.
    	"""

    	path = "/Users/jw598/Cui_group_research/Machine_learning_CE/codes/analysis/EL/ClusterAnalysis/cluster_analysis/data/test"

    	os.chdir(path)

    	with open("cluster_sizes_hist.txt", 'r') as f:

    		for i, line in enumerate(f):

    			if i == 0:
    			    self.assertEqual(line.strip(), "4,4")

    			elif i == 1:
    				self.assertEqual(line.strip(), "8,8")


    def test_cif(self):
        """
        Test function for cif files.
        """

        import glob

        path = "/Users/jw598/Cui_group_research/Machine_learning_CE/codes/analysis/EL/ClusterAnalysis/cluster_analysis/data/test"

        os.chdir(path)

        names = [os.path.basename(x) for x in glob.glob('images_cluster/*')]

        self.assertCountEqual(names, ['image_0_cluster_0_4-4.cif', 'image_0_cluster_1_8-8.cif'])


################################################

if __name__ == '__main__':
    unittest.main()
