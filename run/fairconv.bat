python F:\Users\fetia\Src\FS_NEW\fairseq_latest\train.py ^
F:\Users\fetia\Data\wmt19.db.bt1.tokenized.fi-en.joined --log-interval 5 --no-progress-bar ^
--max-update 30000 --share-all-embeddings --optimizer adam ^
--adam-betas (0.9,0.98) --lr-scheduler inverse_sqrt ^
--clip-norm 0.0 --weight-decay 0.0 ^
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 ^
--min-lr 1e-09 --update-freq 16 --attention-dropout 0.1 --keep-last-epochs 10 ^
--ddp-backend=no_c10d --max-tokens 3584 ^
--lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 ^
--lr-shrink 1 --max-lr 0.001 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 ^
--t-mult 1 --lr-period-updates 70000 ^
--arch lightconv_wmt_en_fr_big --save-dir F:\Users\fetia\Ckpts\wmt19_enfi\fconv_dbbt_fien ^
--dropout 0.1 --attention-dropout 0.1 --weight-dropout 0.1 ^
--encoder-glu 1 --decoder-glu 1 ^