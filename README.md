# foredoc
ForeDoc - MedGemma Radiology

How to run:

git clone https://github.com/alkari/foredoc.git

cd foredoc/

vi dot_env_sample

mv dot_env_sample .env # ADD YOUR KEYS

python3 -m venv foredoc

source foredoc/bin/activate

pip install -r requirements.txt

./start.sh &

fg

deactivate

rm -rf foredoc/
