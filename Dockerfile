FROM ubuntu:18.04
# update apt and get miniconda
RUN apt-get update \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


# install miniconda
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

# run Jupyter install
RUN conda install jupyter

# create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda","run","-n","automl_py37","/bin/bash","-c"]
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install autokeras==0.4.0
RUN apt-get install graphviz graphviz-dev -y
RUN pip install pygraphviz
RUN conda install -c conda-forge python-graphviz

#COPY . ./
RUN apt-get install -y git
RUN git clone https://github.com/heuritech/convnets-keras.git
RUN cd convnets-keras && python setup.py install

RUN pip install git+https://github.com/raghakot/keras-vis.git 

RUN python -m ipykernel install --name automl_py37 --display-name "automl_py37"

SHELL ["/bin/bash","-c"]
RUN conda init
RUN echo 'conda activate automl_py37' >> ~/.bashrc

COPY BioAutoMATED/ ./BioAutoMATED/

EXPOSE 8888                                           
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''","--NotebookApp.password=''"]



