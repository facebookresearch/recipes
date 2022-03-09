# Running

## Locally
torchx run -s local_cwd dist.ddp -j 1x2 --script dlrm_main.py

## To specify arguments
To specify arguments from the main function, add a '--' and then write arguments
e.g:
torchx run -s local_cwd dist.ddp -j 1x2 --script dlrm_main.py \
    -- --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
