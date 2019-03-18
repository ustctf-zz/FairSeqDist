#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer

from operator import itemgetter
import numpy as np
import time


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)
    
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split('::'), task, model_arg_overrides=eval(args.model_overrides))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    if args.decode_source_file is not None:
        print('| [decode] decode from file')
        decode_from_file(models, task, args, use_cuda)
    else:
        print('| [decode] decode from dataset')
        decode_from_dataset(models, task, args, use_cuda)


# TODO: Still need to fix it.
def decode_from_dataset(models, task, args, use_cuda, output_filename=None):
    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    output_filename = output_filename if output_filename is not None else args.decode_output_file
    if output_filename is not None:
        base_filename = output_filename
    else:
        base_filename = args.gen_subset
        if args.num_shards:
            base_filename += "%.2d" % args.shard_id
    decode_filename = _decode_filename(base_filename, args)
    outfile = open(decode_filename, "w")

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    # scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    if args.score_reference:
        translations = translator.score_batched_itr(itr, cuda=use_cuda, timer=gen_timer)
    else:
        translations = translator.generate_batched_itr(
            itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
        )

    wps_meter = TimeMeter()
    for sample_id, src_tokens, target_tokens, hypos in translations:
        # Process input and ground truth
        has_target = target_tokens is not None
        target_tokens = target_tokens.int().cpu() if has_target else None

        # Either retrieve the original sentences or regenerate them from tokens.
        if align_dict is not None:
            src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
            target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
        else:
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            if has_target:
                target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

        if not args.quiet:
            try:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))
            except:
                print('S-{}\t{}'.format(sample_id, src_str.encode('utf-8')))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str.encode('utf-8')))

        # Process top predictions
        for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )

            if not args.quiet:
                try:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                except:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str.encode('utf-8')))
                print('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        hypo['positional_scores'].tolist(),
                    ))
                ))
                print('A-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                ))

            # Score only the top hypothesis
            if has_target and i == 0:
                if align_dict is not None or args.remove_bpe is not None:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    target_tokens = tokenizer.Tokenizer.tokenize(
                        target_str, tgt_dict, add_if_not_exist=True)

        wps_meter.update(src_tokens.size(0))

        num_sentences += 1

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))


def decode_from_file(models, task, args, use_cuda, source_filename=None,
                     target_filename=None, output_filename=None):
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # I/O files
    source_filename = source_filename if source_filename is not None else args.decode_source_file
    target_filename = target_filename if target_filename is not None else args.decode_target_file
    output_filename = output_filename if output_filename is not None else args.decode_output_file
    if output_filename is not None:
        base_filename = output_filename
    else:
        base_filename = source_filename
        if args.num_shards:
            base_filename += "%.2d" % args.shard_id
    decode_filename = _decode_filename(base_filename, args)

    if args.decode_to_file:
        print("| [decode] writing decodes into {}".format(decode_filename))

    # Get sorted input (and reversed)
    sorted_inputs, sorted_keys, sorted_targets = _get_sorted_inputs(
        source_filename, args.num_shards, args.delimiter, target_filename, args.shard_id)

    # Build input iterator
    src_tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in sorted_inputs]
    src_sizes = np.array([t.numel() for t in src_tokens])
    tgt_tokens = [
        tokenizer.Tokenizer.tokenize(tgt_str, tgt_dict, add_if_not_exist=False).long()
        for tgt_str in sorted_targets] if sorted_targets is not None else None
    tgt_sizes = np.array([t.numel() for t in tgt_tokens]) if tgt_tokens is not None else None
    print('| loading {} examples, {} tokens'.format(len(sorted_inputs), sum(src_sizes)))

    # Load dataset (possibly sharded)
    dataset = data.LanguagePairDataset(
        src_tokens, src_sizes, src_dict, tgt_tokens, tgt_sizes, tgt_dict, shuffle=False)
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )

    if use_cuda:
        translator.cuda()

    num_sentences = 0

    if args.score_reference:
        translations = translator.score_batched_itr(itr, cuda=use_cuda, timer=gen_timer)
    else:
        translations = translator.generate_batched_itr(
            itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
        )

    decodes = dict()
    eval_scores = dict()
    sids = []
    wps_meter = TimeMeter()
    start = time.perf_counter()
    for sample_id, src_tokens, target_tokens, hypos in translations:
        # Process input and ground truth
        has_target = target_tokens is not None
        target_tokens = target_tokens.int().cpu() if has_target else None

        # Either retrieve the original sentences or regenerate them from tokens.
        if align_dict is not None:
            src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
            target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
        else:
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            if has_target:
                target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

        if not args.quiet:
            try:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))
            except:
                print('S-{}\t{}'.format(sample_id, src_str.encode('utf-8')))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str.encode('utf-8')))

        # Process top predictions
        for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            if i == 0:
                if args.score_reference:
                    eval_scores[sample_id.tolist()] = hypo['score']
                else:
                    decodes[sample_id.tolist()] = hypo_str

            if not args.quiet:
                try:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                except:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str.encode('utf-8')))
                print('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        hypo['positional_scores'].tolist(),
                    ))
                ))

                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

            if args.remain_nbest and i != 0:
                decodes[sample_id.tolist()] += "{}{}".format(args.delimiter, hypo_str)

        wps_meter.update(src_tokens.size(0))

        num_sentences += 1
        if args.quiet and num_sentences % 5000 == 0:
            print("| {} / {} sentences decoded ({}), time: {:.3f}s".format(num_sentences, len(sorted_inputs), len(decodes), time.perf_counter() - start))

    used_time = time.perf_counter() - start
    print("| Used time:" + repr(used_time))
    print("| Average time:" + repr(used_time / len(sorted_inputs)))

    if args.decode_to_file or args.score_reference:
        outfile = open(decode_filename, "w")
        print("| [{}] writing decodes into {}".format('eval' if args.score_reference else 'decoding', decode_filename))
        # print(sids)
        for index in range(len(sorted_inputs)):
            try:
                outfile.write("{}{}".format(eval_scores[sorted_keys[index]] if args.score_reference else decodes[sorted_keys[index]] , args.delimiter))
            except:
                outfile.write("{}{}".format(eval_scores[sorted_keys[index]] if args.score_reference else decodes[sorted_keys[index]].encode('utf-8'), args.delimiter))
        outfile.close()

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))


def _get_sorted_inputs(filename, num_shards=1, delimiter="\n",
                       targets_filename=None, worker_id=None):
    print("| getting sorted inputs")
    # read file and sort inputs according them according to input length.
    if num_shards > 1:
        assert worker_id == None
        source_filename = filename + ("%.2d" % worker_id)
    else:
        source_filename = filename
    print("| [src] {}".format(source_filename))

    # with open(source_filename, "r") as f:
    with open(source_filename, "r", encoding="utf-8") as f:
        text = f.read()
        records = text.split(delimiter)
        inputs = [record.strip() for record in records]
        # Strip the last empty line.
        if not inputs[-1]:
            inputs.pop()

    if targets_filename is not None:
        if num_shards > 1:
            targets_filename += "%.2d" % worker_id
        # with open(targets_filename, "r") as f:
        with open(targets_filename, "r", encoding="utf-8") as f:
            text = f.read()
            records = text.split(delimiter)
            targets = [record.strip() for record in records]
            if not targets[-1]:
                targets.pop()
        assert len(targets) == len(inputs)
        print("| [trg] {}".format(targets_filename))

    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=itemgetter(1), reverse=True)
    # We'll need the keys to rearrange the inputs back into their original order
    sorted_keys = {}
    sorted_inputs = []
    sorted_targets = None if targets_filename is None else []
    for new_index, (orig_index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[orig_index])
        if targets_filename is not None:
            sorted_targets.append(targets[orig_index])
        sorted_keys[orig_index] = new_index
    return sorted_inputs, sorted_keys, sorted_targets


def _decode_filename(base_filename, args):
    return base_filename
    '''
    return "{base}{arch}.beam{beam}.lpen{lpen}.decodes{bpe}{dedup}{idx}".format(
        base=base_filename,
        arch="."+args.model_arch if args.model_arch is not None else "",
        beam=str(args.beam),
        lpen=str(args.lenpen),
        bpe=".atat" if args.remove_bpe else "",
        dedup=".dedup" if args.dedup else "",
        idx=".index" if args.decode_to_index else "")
    '''

def add_user_extra_generation_args(parser, seq=False):
    group = parser.add_argument_group('Extra generation args')
    # seq_translate
    if seq:
        group.add_argument('--generate-code-path', default="/var/storage/shared/msrmt/fetia/yiren/fairseq/",
                           type=str, metavar="PATH", help="path to generate.py")
        group.add_argument('--ckpt-dir', metavar='PATH',
                           help='path(s) to model file(s), colon separated')
        group.add_argument('--initial-model', default=None, metavar='FILE',
                           help='The first model to be tested')
    # decode from file
    group.add_argument('--decode-source-file', default=None, type=str, metavar='FILE')
    group.add_argument('--decode-target-file', default=None, type=str, metavar='FILE')
    group.add_argument('--decode-output-file', default=None, type=str, metavar='FILE')
    group.add_argument('--decode-to-file', action="store_true")
    group.add_argument("--decode-to-index", action="store_true")
    group.add_argument('--delimiter', default="\n", type=str)
    group.add_argument('--dedup', action="store_true")
    group.add_argument('--remain-nbest', action="store_true")

    # eval from file
    group.add_argument('--source-file', default=None, type=str, metavar='FILE')
    group.add_argument('--target-file', default=None, type=str, metavar='FILE')
    group.add_argument('--score-file', default=None, type=str, metavar='FILE')

    group.add_argument('--model-arch', default=None, type=str, metavar="MODELARCH")

    return group


if __name__ == '__main__':
    parser = options.get_generation_parser()
    add_user_extra_generation_args(parser)
    args = options.parse_args_and_arch(parser)
    print(args)
    main(args)
