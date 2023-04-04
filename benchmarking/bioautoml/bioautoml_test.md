Downloaded from the following [link](https://github.com/Bonidia/BioAutoML).

Installed the environment in the following way:

`git clone https://github.com/Bonidia/BioAutoML.git BioAutoML
cd BioAutoML
git submodule init
git submodule update
conda env create -f BioAutoML-env.yml -n bioautoml
conda activate bioautoml`

Note I did need to use an Ubuntu 18.04 virtual machine as this was incompatible with Mac OS.

First I confirmed that the test case study employed by the BioAutoML authors worked.

`cd BioAutoML/

python BioAutoML-feature.py -fasta_train Case\ Studies/CS-I-A/E_coli/train/rRNA.fasta Case\ Studies/CS-I-A/E_coli/train/sRNA.fasta -fasta_label_train rRNA sRNA -fasta_test Case\ Studies/CS-I-A/E_coli/test/rRNA.fasta Case\ Studies/CS-I-A/E_coli/test/sRNA.fasta -fasta_label_test rRNA sRNA -output test_directory

python BioAutoML+iFeature-protein.py -fasta_train MathFeature/Case\ Studies/CS-V/anticancer.fasta MathFeature/Case\ Studies/CS-V/non.fasta -fasta_label_train anticancer non -output cancertest`

To benchmark the test datasets, I ran the following:

`python BioAutoML-feature.py -fasta_train ../classification_small_synthetic_posseqs.fasta ../classification_small_synthetic_negseqs.fasta -fasta_label_train pos neg -output small_synthetic

python BioAutoML-feature.py -fasta_train ../classification_large_synthetic_posseqs.fasta ../classification_large_synthetic_negseqs.fasta -fasta_label_train pos neg -output large_synthetic

python BioAutoML-feature.py -fasta_train ../data/classification_toeholds_posseqs.fasta ../data/classification_toeholds_negseqs.fasta -fasta_label_train pos neg -output toeholds

python BioAutoML-feature.py -fasta_train ../data/classification_hollerer_rbs_train_posseqs.fasta ../data/classification_hollerer_rbs_train_negseqs.fasta -fasta_label_train pos neg -output full_hollerer_rbs_train

python BioAutoML-feature.py -fasta_train ../data/classification_hollerer_rbs_mediumtrain_posseqs.fasta ../data/classification_hollerer_rbs_mediumtrain_negseqs.fasta -fasta_label_train pos neg -output hollerer_rbs_mediumtrain

python BioAutoML+iFeature-protein.py -fasta_train ../data/classification_train_NO_J_peptides_posseqs.fasta ../data/classification_train_NO_J_peptides_negseqs.fasta -fasta_label_train pos neg -output no_j_peptides_more_cpu -n_cpu 8`
