#!sh

rm -rf games/*.txt

python mcts_v_randombot.py > games/1.txt &
python abprune_v_mcts.py > games/2.txt &
python abprune_v_randombot.py > games/3.txt