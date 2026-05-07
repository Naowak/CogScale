echo "Running baseline experiments with all models with 100k parameters..."

echo "Running LSTM model..."
python run.py \
    --tasks cross_situation \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type lstm transformer_decoder \
    --output cross_situation_lstm_transformer_decoder.csv

python run.py \
    --tasks cross_situation \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type gru transformer_encdec \
    --output cross_situation_gru_transformer_encdec.csv


python run.py \
    --tasks cross_situation \
    --difficulties small medium \
    --sizes 1000 10000 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cpu \
    --dtype float32 \
    --model_type lstm gru transformer_decoder transformer_encdec \
    --output cross_situation.csv

echo
echo "Running GRU model..."
python run.py \
    --tasks all \
    --difficulties small \
    --sizes 10000 \
    --seeds 10 \
    --epochs 200 \
    --device cpu \
    --dtype float32 \
    --model_type esn \
    --output esn_10k_small.csv

echo
echo "Running Transformer Encoder-Decoder model..."
python run.py \
    --tasks all \
    --difficulties medium \
    --sizes 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:0 \
    --dtype float32 \
    --model_type transformer_encdec \
    --output transformer_encdec_100k_medium.csv


echo
echo "Running Transformer Decoder model..."
python run.py \
    --tasks all \
    --difficulties medium \
    --sizes 100000 \
    --seeds 10 \
    --epochs 200 \
    --device cuda:1 \
    --dtype float32 \
    --model_type transformer_decoder \
    --output transformer_decoder_100k_medium.csv