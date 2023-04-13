# BioAutoMATED Installation Guide

There are two options for code installation: 1) Docker and 2) GitHub. We recommend using Docker because the package installations are automatically handled and guaranteed to work after Docker is successfully installed. For some users, you may wish to download the packages yourself from GitHub into a conda virtual environment. If you decide to go with this installation option, please ensure that your computer or environment can install TensorFlow v1.13.1 (with or without GPUs). Unfortunately, the new Macs with M1 chips cannot install this version of TensorFlow.

## Option #1: Docker Installation (recommended, no package installations required after Docker)

1.	Install Docker.
    * Download the app for your specific system by following the instructions on the Docker installation page. The current instructions are reproduced in part here for ease of use. Follow the Docker instructions for your specific system. 
    * For Macs with Intel Chip: macOS must be version 11 or newer. That is, Big Sur, Monterey, or Ventura. We recommend upgrading to the latest version of macOS. At least 4 GB of RAM is required.
    * For Macs with Apple Chip: macOS must be Ventura.
    * For Windows with WSL 2 backend: The requirements are as follows: The following hardware prerequisites are required to successfully run WSL 2 on Windows 10 or Windows 11:
        * 64-bit processor with Second Level Address Translation (SLAT)
        * 4GB system RAM
        * BIOS-level hardware virtualization support must be enabled in the BIOS settings. 
        * Windows 11 64-bit: Home or Pro version 21H2 or higher, or Enterprise or Education version 21H2 or higher. Windows 10 64-bit: Home or Pro 2004 (build 19041) or higher, or Enterprise or Education 1909 (build 18363) or higher.
        * Enable the WSL 2 feature on Windows. For detailed instructions, refer to the Microsoft documentation.
    * For example, for MacOS running the Apple chip, you should double click Docker.dmg and then drag the Docker.dmg into the applications folder. Then, double click Docker.app in the Applications folder to start Docker. Docker should now be open on your computer at this point. 
    * For Linux environments, we used the following guidelines for installing Docker on a Linux instance (here, Ubuntu 18.04 LTS). Reproducing the code below for reference.
        * `sudo apt-get update`
        * `sudo apt-get install \ 
        ca-certificates \
        curl \
        gnupg \
        lsb-release`
        * `sudo mkdir -m 0755 -p /etc/apt/keyrings`
        * `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg`
        * `echo \ 
         "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \ 
         "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | 
         \ sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`
        * `sudo apt-get update`
        * `sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`

2.	For M1/2 Macs, we will need to make a change to default parameters because the performance of Docker on the new Apple silicon chips is still under development. 
     * Some details for those interested follow. If you are not interested, please go directly to b) to skip to implementation. The current situation: As we cannot support an aarch64/ARM64 build of the BioAutoMATED Docker container, we need to support the x86/AMD64 architecture to run on M1/2 Macs. Prior to January 2023, the best way to run x86/AMD64 architectures on M1/2 Macs was via emulation, which has extremely slow performance for BioAutoMATED and many other packages. This is a documented issue with Docker on M1/2 Macs, with more details here. In January 2023, an experimental feature was enabled to run x64/AMD64 architectures on M1/2 Macs via Rosetta virtualization. This is not as fast as natively running BioAutoMATED (again, consistent with reports from other packages) but is a great enhancement from the previous emulation option. We recommend using this experimental feature for best performance and we will be closely following updates on this problem to suggest alternatives if they are developed.
    * Go into Settings/General/ and select Use Virtualization Framework. M1/2 Mac users should go into the Features in Development selection and select “Use Rosetta for x86/amd64 emulation on Apple Silicon”. Note that to see this option, you must have Mac OS Ventura. You will need to restart Docker for changes to apply. A more detailed guide to these instructions can be found here. 

3.	Next, open terminal if it is not already open and navigate to a folder where you would like to work out of. For example, this may be your desktop: cd Desktop/. The next part of these instructions is adapted from the Docker installation instructions from the PyModulon package here.

4.	Run the following command to test your Docker installation: “docker run hello-world” You should see something along the lines of “Hello from Docker! This message shows that your installation appears to be working correctly.” Do not proceed if you do not have this running successfully!
    * Before proceeding to the next step, make sure you do not have anything running on your IP address port 8888. We will be using this port for our Jupyter notebook.

5.	Finally, you are ready to download the Docker repository and get started.
    * Pull the repository: `docker pull jackievaleri/bioautomated:v5`
    * After this stage, there should be a message saying where the image was downloaded to, for example “docker.io/jackievaleri/bioautomated:v5”. 
    * You can then start up the docker container: `docker run -dp 8888:8888 --shm-size 16G docker.io/jackievaleri/bioautomated:v5 [or the relevant location]`
        * For M1/2 Macs, you may see an error saying: “The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)”. This relates to the issue raised in point #2. You can still proceed.
        * We are using this `shm-size` flag to make AutoKeras play nicely with the Docker container, as detailed here.
    * You can check what containers are running with the command `docker ps -a`.
    * Now, you should be able to copy your IP address plus the port number to access the code. Your IP address is localhost if you are running this guide on your machine, or your external IP address if you are running this on a virtual machine. For example, copy `localhost:8888` into your Web browser.
    * Then, open up the 01_BioAutoMATED_Small_System_Test_START_HERE.ipynb. Depending on where you are, you may need to first navigate into a folder called BioAutoMATED/. You should be able to run everything.
 
## Option #2: GitHub Download & Conda Installation

1.	First, download the repository from GitHub at: https://github.com/jackievaleri/BioAutoMATED or run the following command:
    * git clone https://github.com/jackievaleri/BioAutoMATED.git BioAutoMATED

2.	If you do not already have conda, install Anaconda and add it to your path. For example, 
    * `wget http://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh`
    * `sudo bash Anaconda3-5.3.1-Linux-x86_64.sh`
    * `export PATH=~/anaconda3/bin:$PATH`
    * `vim ~/.bashrc` and add `“export PATH=~/anaconda3/bin:$PATH”` to the last line
    * `source ~/.bashrc` or reboot
    * Check that the install worked by trying: `conda –version`. There should be an output with your conda version.

3.	Install Python 3.7 if you do not already have it: conda install -c anaconda python=3.7

4.	Create and activate a virtual environment called automl_py37 with python 3.7 from the environment.yml file.
    * `conda env create -f environment.yml`
    * `conda activate automl_py37`

5.	Finish with the last few installations that do not play well with conda:
    * Note: you should replace instances of `pip` and `python` with `pip3` and `python3`, respectively, if you have older installations of python2 on your computer. This will force the python3 path to be used.
    * `pip install -r requirements.txt`
    * `pip install autokeras==0.4.0`
        * Note, you may see this error: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. deepswarm 0.0.10 requires scikit-learn==0.20.3, but you have scikit-learn 0.20.2 which is incompatible.`
       * If you have this issue, you can use this instead: `pip3 install autokeras==0.4.0 --no-deps`
       * Go ahead and proceed – scikit-learn incompatibilities have been factored in.
    * `apt-get install graphviz graphviz-dev -y`
       * For Mac, okay to use: `brew install graphviz`
       * Note, you may see this error: `Error: Failed to download resource "libpng"`
       * If you see this error, you can use `pip3 install graphviz` instead.
    * `pip install pygraphviz`
    * `conda install -c conda-forge python-graphviz`
    * `git clone https://github.com/heuritech/convnets-keras.git`
    * `cd convnets-keras`
    * `python setup.py install`
    * `cd ..`
    * `pip install git+https://github.com/raghakot/keras-vis.git`

6.	Deactivate the conda environment and add the environment to your ipykernel: 
    * `conda deactivate`
    * `python -m ipykernel install --user --name=automl_py37`

7.	At this point, you should be able to use the command jupyter notebook, which should launch the Jupyter window in your web browser. Open up the notebook 01_BioAutoMATED_Small_System_Test_START_HERE.ipynb and select the kernel: automl_py37. 
    * Note: Tensorflow 1.13.1 does not currently run on Macs with M1 chips.

8.	If you prefer to run BioAutoMATED in command line, navigate to the folder called BioAutoMATED. Then run the following:
    * `conda activate automl_py37`
    * `python main_classes/wrapper.py -task binary_classification -data_folder ./clean_data/clean/ -data_file small_synthetic.csv -sequence_type nucleic_acid -model_folder ./exemplars/test_synthetic_nucleic_acids/models/ -output_folder ./exemplars/test_synthetic_nucleic_acids/outputs/ -verbosity 1 -input_col seq -target_col positive_score -max_runtime_minutes 10 -num_folds 2 -num_final_epochs 10 -num_generations 5 -population_size 5`

    * This replicates the output of the small system test described in Step 7 above. For a full list of arguments and their meaning, please reference either the .ipynb notebook or the run_bioautomated function in the wrapper.py file.
    * Also, please note that you will have to make the test_synthetic_nucleic_acids/ folder in addition to the models/ and outputs/ subfolders if not already present.

9.	To close the notebook, press Ctrl+C in terminal. All changes made to files in your current directory are saved to your local machine.

 
## Troubleshooting:

* Problem: When I pull the Docker repository, I get a “permission denied” or “unauthorized: incorrect username or password” error.
* Solution: Login to your Docker account before pulling, for example with the command “docker login –username [your username]”.



* Problem: The page buffers but never loads when I enter the address for the Jupyter notebook.
* Solution: This problem could be caused by a variety of issues, but we often see it when there is an issue connecting with the port. Check that port 8888 is available on your machine. If you are using a platform like Google Cloud Platform to make a virtual machine, it is also important to make a firewall rule for port 8888 if you are using a virtual machine on Google Cloud Platform, for instance. You may not want to use the 0.0.0.0/0 (open to everything) IP range if you have not specified which instances this firewall rule is applicable for, or else your system will be open to anyone.
  
  
  
* Problem: I have an M1 or M2 Mac and Docker seems to be running slowly.
* Solution: We are working on optimizing the build for Macs with M1/2 Apple chips. Please see point #2 under the Docker installation instructions. Unfortunately, this is a wider problem with the new Apple silicon chips and we will be closely following developments in this space.
