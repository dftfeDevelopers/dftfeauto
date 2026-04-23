#!/usr/bin/env bash
# run_make_invParams.sh
# Runs the Python script that generates invParams.prm

module load python

export invdft_exe="$HOME/softwares/invDFT_projectToSlater/invdft/build_publicDftfe/release/real/invDFT_exe"
python3 extractMolecularData.py hbci.inp 

for i in *_basis.bas;
do 
	echo $i; 
	python3 convert_basis.py $i; 
done;

python3 compute_pyscf_1rdm.py > output_pyscf_lda

cat output_pyscf_lda

for i in *_basis.bas;
do
        echo $i;
        python3 basis_d_to_e.py $i;
done;

mkdir invDFT_calc
for i in *_invdft.bas;
do
        echo $i;
        cp $i invDFT_calc
done;
cd invDFT_calc
cp ../createCoordFiles.py .
cp ../createGSParams.py .
cp ../createInvParams.py .
cp ../createDomainVectors.py .
cp ../molecule.xyz . 
cp ../writeJobScript.py .
python3 createCoordFiles.py .
python3 createGSParams.py .
python3 createInvParams.py
python3 createDomainVectors.py
python3 writeJobScript.py
sbatch job.slurm
cp ../alphaPao DensityMatrix_HBCI
cp ../DensityMat_lda_pyscf  DensityMatrix_LDA
cp ../ovlpAO  SMatrix

cp "$invdft_exe" .
