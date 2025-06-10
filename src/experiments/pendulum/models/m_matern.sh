source ~/miniconda3/miniconda3.sh
conda activate physs_gp

cd ~/git/PHYSS_GP/src/experiments/pendulum/models

python m_matern.py -i 0
python m_matern.py -i 1
python m_matern.py -i 2
python m_matern.py -i 3
python m_matern.py -i 4
python m_matern.py -i 5
python m_matern.py -i 7