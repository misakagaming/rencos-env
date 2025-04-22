import os
import sys
import time

def main(opt, seed, mode=2):
    if opt == 'preprocess':
        command = "python preprocess.py -train_src source_alt/java/train/train.spl.src \
                        -train_tgt source_alt/java/train/train.txt.tgt \
                        -valid_src source_alt/java/valid/valid.spl.src \
                        -valid_tgt source_alt/java/valid/valid.txt.tgt \
                        -save_data source_alt/%s/preprocessed/baseline_spl \
                        -src_seq_length 10000 \
                        -tgt_seq_length 10000 \
                        -src_seq_length_trunc %d \
                        -tgt_seq_length_trunc %d \
                        -seed %d" % (lang, src_len, tgt_len, seed)
        os.system(command)
    elif opt == 'train':
        command = "python train.py -word_vec_size 256 \
                        -layers 1 \
                        -rnn_size 512 \
                        -rnn_type LSTM \
                        -src_word_vec_size 512 \
                        -tgt_word_vec_size 512 \
                        -word_vec_size 512 \
                        -pre_word_vecs_enc /custom_embeddings/concat_weights.pt \
                        -global_attention mlp \
                        -data source_alt/%s/preprocessed/baseline_spl \
                        -save_model models/%s/baseline_spl \
                        -gpu_ranks 0 \
                        -batch_size 32 \
                        -optim adam \
                        -learning_rate 0.001 \
                        -dropout 0 \
                        -encoder_type brnn \
                        -seed %d" % (lang, lang, seed)
        os.system(command)
    elif opt == 'retrieval':
        print('Syntactic level...')
        command1 = "python syntax.py %s" % lang
        os.system(command1)
        print('Semantic level...')
        batch_size = 32 if lang == 'python' else 16
        command2 = "python translate_alt.py -model models/%s/baseline_spl_step_100000.pt \
                        -src source_alt/java/train/train.spl.src \
                        -output source_alt/%s/output/train.out \
                        -batch_size %d \
                        -gpu 0 \
                        -fast \
                        -max_sent_length %d \
                        -refer 0 \
                        -lang %s \
                        -search 2 \
                        -seed %d" % (lang, lang, batch_size, src_len, lang, seed)
        os.system(command2)
        command3 = "python translate_alt.py -model models/%s/baseline_spl_step_100000.pt \
                        -src source_alt/java/test/test.txt.src \
                        -output source_alt/%s/test/test.ref.src.1 \
                        -batch_size 32 \
                        -gpu 0 \
                        -fast \
                        -max_sent_length %d \
                        -refer 0 \
                        -lang %s \
                        -search 2 \
                        -seed %d" % (lang, lang, src_len, lang, seed)
        os.system(command3)
        print('Normalize...')
        command4 = "python normalize.py %s" % lang
        os.system(command4)
    elif opt == 'translate':
        command = "python translate_alt.py -model models/%s/baseline_spl_step_100000.pt \
                    -src source_alt/java/test/test.txt.src \
                    -output source_alt/%s/output/test.out \
                    -min_length 3 \
                    -max_length %d \
                    -batch_size 32 \
                    -gpu 0 \
                    -fast \
                    -max_sent_length %d \
                    -refer %d \
                    -lang %s \
                    -beam 5" % (lang, lang, tgt_len, src_len, mode, lang)
        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    lang = sys.argv[2]
    assert option in ['preprocess', 'train', 'retrieval', 'translate', 'all']
    assert lang in ['python', 'java']
    if lang == 'python':
        src_len, tgt_len = 100, 50
    elif lang == 'java':
        src_len, tgt_len = 300, 30
    else:
        print("Unsupported Programming Language:", lang)
    if option == 'all':
        main('preprocess')
        main('train')
        main('retrieval')
        main('translate')
    else:
        if option == 'translate':
            mode = int(sys.argv[3])
            seed = int(sys.argv[4])
            main(option, seed, mode)
        else:
            seed = int(sys.argv[3])
            main(option, seed)
