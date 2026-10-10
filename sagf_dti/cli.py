"""Commands for the original Gaussian SAGF-DTI implementation."""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train and reproduce the original Gaussian SAGF-DTI model.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate', 'validate-data'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--data-root', required=True)
        if name != 'validate-data':
            p.add_argument('--output', required=True)
            p.add_argument('--device', default=None)
            p.add_argument('--disable-host-pinning', action='store_true',
                           help='Record a transfer-only pin_memory=False override for constrained hosts.')
        if name == 'train':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--max-batches', type=int, default=None)
            p.add_argument('--require-paper-config', action='store_true')
        elif name == 'evaluate':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--split', choices=('val', 'test'), default='test')
        else:
            p.add_argument('--split', choices=('train', 'val', 'test'), action='append')
            p.add_argument('--inspect-features', action='store_true')
    p = sub.add_parser('summarize')
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            from .train import train
            result = train(args.config, args.data_root, args.output, device=args.device,
                           dry_run=args.dry_run, max_batches=args.max_batches,
                           require_paper_config=args.require_paper_config,
                           disable_host_pinning=args.disable_host_pinning)
        elif args.command == 'evaluate':
            from .evaluate import evaluate_checkpoint
            result = evaluate_checkpoint(args.config, args.data_root, args.checkpoint, args.output,
                                         split=args.split, device=args.device,
                                         disable_host_pinning=args.disable_host_pinning)
        elif args.command == 'validate-data':
            from .config import resolve_config
            from .data import validate_data
            result = validate_data(resolve_config(args.config, args.data_root),
                                   splits=tuple(args.split or ('train', 'val')),
                                   inspect_features=args.inspect_features)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['valid'] else 1
        else:
            from .report import summarize_runs
            result = summarize_runs(args.runs, args.output)
        # Completion manifests contain hashes for many caches; print a concise result.
        print(json.dumps({k:v for k,v in result.items() if k not in ('verified_files', 'source_sha256',
                                                                  'input_sha256', 'completed_runs')},
                         ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'SAGF-DTI: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
