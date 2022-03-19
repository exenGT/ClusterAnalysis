import unittest
import os

class TestAnalysisMethods(unittest.TestCase):

    def test_run(self):

        from cluster_analysis.cluster import analyze

        ## 1. need to have relative path
        ## 2. write the output into a temp folder
        analyze(args_list=['--path', "./",
                           '--num_dump', "1",
                           '--dump_dir', "dump_file_test",
                           '--info_file', "info_file.txt",
                           '--salt_unit_file', "LiFSI.xyz"])

        ## test the cif files
        import glob

        path = "./"

        os.chdir(path)

        names = [os.path.basename(x) for x in glob.glob('images_cluster/*')]

        ## glob.glob may return random order
        self.assertCountEqual(names, ['image_0_cluster_0_4-4.cif', 'image_0_cluster_1_8-8.cif'])

        ## test the hist file
        with open("cluster_sizes_hist.txt", 'r') as f:

            for i, line in enumerate(f):

                if i == 0:
                    self.assertEqual(line.strip(), "4,4")

                elif i == 1:
                    self.assertEqual(line.strip(), "8,8")

        ## write some other tests

################################################

if __name__ == '__main__':
    unittest.main()
