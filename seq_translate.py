from generate import main as single_model_main

import os, sys, subprocess
import time, sys
import re
from fairseq import options

def obtain_sys_argv():
    def if_illegal_args(str_check):
        illegal_args_names = ['--ckpt', '--initial', '--generate', '--decokenizer', '--sacre-test']
        return any([x in str_check for x in illegal_args_names])
    sys_args = ' '.join([x for (idx, x) in enumerate(sys.argv[1:]) if not if_illegal_args(x) and not if_illegal_args(sys.argv[idx])])
    return sys_args

def main(args):
    print(args)
    sys.stdout.flush()
    files = [os.path.join(args.ckpt_dir, x) for x in os.listdir(args.ckpt_dir) if x.endswith('.pt')]
    files.sort(key=lambda x: os.path.getmtime(x))

    start_idx = 0
    if args.initial_model is not None and args.initial_model is not "":
        initial_model = os.path.join(args.ckpt_dir, args.initial_model)
        for (idx, file) in enumerate(files):
            if file == initial_model:
                start_idx = idx
                print('Staring from ckpt {}'.format(file))

    bleu_ptn = 'BLEU4\s=\s([\d\.]+?),'
    sacre_bleu_ptn = 'BLEU\+case.+?=\s([\d\.]+?)\s.+'
    results = {}
    for x in range(start_idx, len(files)):
        ckpt_file = files[x]
        args.path = ckpt_file
        if not ckpt_file.endswith(".pt"):
            continue
        print('Scoring ckpt {}'.format(ckpt_file))
        sys.stdout.flush()
        #Note here, simply calling single_model_main will bring mysterious memory error, so use bruteforce calling instead
        decode_out_file = '{}/tmp.txt'.format(args.ckpt_dir)
        code_file = "generate.py" if args.sacre_test_set == "wmt18" else "generate_v2.py"

        command = 'python {}/{} --path {} ' \
                  '--output-file {} ' \
                  '{} ' \
                  '{}'.format(args.generate_code_path, code_file, os.path.join(args.ckpt_dir, ckpt_file),
                              decode_out_file,
                            '--decode-source-file {}/{}'.format(args.store_testfiles_path, '{}.test.en-fi.{}'.format(args.sacre_test_set, args.source_lang)) if args.sacre_test_set != 'wmt18' else '',
                            obtain_sys_argv())
        print(command)
        sys.stdout.flush()
        subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE).communicate()
        time.sleep(15)
        pl_process = subprocess.Popen(
            'cat {} | perl {} -l {} | sacrebleu -t {} -l {}-{} -w 2 '.
                format(decode_out_file, args.decokenizer, args.target_lang, args.sacre_test_set, args.source_lang, args.target_lang),
            shell=True,
            stdout=subprocess.PIPE)

        pl_output = pl_process.stdout.read()

        os.remove(decode_out_file)
        bleu_match = re.search(sacre_bleu_ptn, str(pl_output))
        if bleu_match:
            bleu_score = bleu_match.group(1)
            filename = os.path.basename(ckpt_file)
            print(filename, bleu_score)
            sys.stdout.flush()
            results[filename] = bleu_score
            sys.stdout.flush()

    print('\n'.join(['{}\t{}'.format(x, y) for (x,y) in results.items()]))

def add_user_extra_generation_args(parser):
    parser.add_argument('--generate-code-path', default="/var/storage/shared/msrmt/fetia/yiren/fairseq/",
                       type=str, metavar="PATH", help="path to generate.py")
    parser.add_argument('--store-testfiles-path', default="/blob/fetia/Data/test_previous",
                        type=str, metavar="PATH", help="path to store test files, including wmt15 to wmt18")
    parser.add_argument('--decokenizer', default="/blob/fetia/Src/mosesdecoder/scripts/tokenizer/detokenizer.perl",
                        type=str, metavar="PATH", help="path to detokenizer")
    parser.add_argument('--ckpt-dir', metavar='PATH',
                       help='path(s) to model file(s), colon separated')
    parser.add_argument('--initial-model', default=None, metavar='FILE',
                       help='The first model to be tested')
    parser.add_argument('--sacre-test-set', default='wmt18', metavar='STRING',
                       help='data set to test the sacre bleu on')

if __name__ == '__main__':
    parser = options.get_generation_parser()
    add_user_extra_generation_args(parser)

    args = options.parse_args_and_arch(parser)
    main(args)