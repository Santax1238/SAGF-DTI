"""Profile the frozen paper computation on actual longest cached proteins."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .artifacts import atomic_json, environment_record, input_hashes, sha256, source_hashes, utc_now


def indexed_cuda_device(device, *, current_index):
    import torch
    parsed = torch.device(device)
    if parsed.type != 'cuda':
        raise ValueError('The paper memory benchmark requires a CUDA device')
    return torch.device('cuda', parsed.index if parsed.index is not None else current_index)


def profile_loader_options(loader, *, disable_host_pinning=False):
    kwargs = {'num_workers': loader.num_workers, 'collate_fn': loader.collate_fn,
              'pin_memory': False if disable_host_pinning else loader.pin_memory}
    if loader.num_workers:
        kwargs.update(persistent_workers=loader.persistent_workers, prefetch_factor=loader.prefetch_factor)
    return kwargs


def gradient_group_totals(named_gradients):
    """Observe actual GaussianNodeFusion variance rows, independently of means."""
    totals = {'dynamic': 0., 'alpha': 0., 'variance': 0.}
    for name, grad in named_gradients:
        if any(token in name for token in ('dynamic_condition_', 'dynamic_property_proj',
                                          'esmdance35_proj', 'motion_encoder', 'conformation_encoder')):
            totals['dynamic'] += float(grad.float().abs().sum().cpu())
        if 'alpha_head' in name:
            totals['alpha'] += float(grad.float().abs().sum().cpu())
        if any(token in name for token in ('esm_stats.', 'dyn_stats.')):
            totals['variance'] += float(grad[grad.shape[0] // 2:].float().abs().sum().cpu())
    return totals


def longest_pair_indices(rows, validation, batch_size):
    """Select distinct CSV rows by actual graph node count, keeping tie order."""
    if not validation.get('valid') or not validation.get('inspect_features'):
        raise ValueError('A complete inspected graph validation is required')
    if len(rows) < batch_size:
        raise ValueError(f'Need {batch_size} actual pairs, found {len(rows)}')
    graphs = validation['features']['graphs']
    lengths = []
    for index, target in enumerate(rows['sequence_hash'].astype(str)):
        nodes = graphs.get(target, {}).get('nodes')
        if not isinstance(nodes, int) or nodes <= 0:
            raise ValueError(f'Missing inspected graph length for {target}')
        lengths.append((index, nodes))
    return [index for index, _ in sorted(lengths, key=lambda item: (-item[1], item[0]))[:batch_size]]


def assert_benchmark_ready(profile):
    checks = {
        'complete': profile.get('complete') is True, 'passed': profile.get('passed') is True,
        'CUDA device': str(profile.get('device', '')).startswith('cuda'),
        'original batch32': profile.get('batch_size') == 32,
        'original validation64': profile.get('eval_batch_size') == 64,
        'original FP16 AMP': profile.get('amp_dtype') == 'float16' and profile.get('grad_scaler') is True,
        'original recipe': profile.get('original_paper_recipe') is True,
        'optimizer updates': profile.get('optimizer_steps', 0) >= 2,
        'finite losses': profile.get('finite_losses') is True,
        'finite gradients': profile.get('finite_gradients') is True,
        'longest actual pairs': profile.get('longest_pairs_verified') is True,
        'contrastive': profile.get('contrastive_enabled') is True,
        'auxiliary': profile.get('auxiliary_enabled') is True,
        'hypergradient': profile.get('hypergradient_enabled') is True and profile.get('hypergradient_updated') is True,
    }
    for name in ('dynamic', 'alpha', 'variance'):
        checks[f'{name} gradients'] = profile.get('gradient_groups_nonzero', {}).get(name) is True
    memory = profile.get('peak_reserved_gib', math.inf)
    limit = min(float(profile.get('max_reserved_gib', 10.5)), 10.5)
    checks['GPU memory'] = isinstance(memory, (int, float)) and math.isfinite(memory) and 0 < memory < limit
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError('Paper benchmark did not pass: ' + ', '.join(failures))
    return profile


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def benchmark(config_path, data_root, output, *, device='cuda', max_reserved_gib=10.5, steps=3,
              disable_host_pinning=False):
    """Run original full batch32 forward/backward/loss/HG and validation64.

    This is a memory and correctness diagnostic, not a trained result. The
    original ``train_one_epoch`` function is called without numerical edits.
    Gradient observations add synchronizations, so timings are diagnostic.
    """
    import torch
    from torch.utils.data import DataLoader
    from .config import assert_paper_config, resolve_config
    from .data import validate_data
    from .legacy import load_trainer

    if steps < 2 or not 0 < max_reserved_gib <= 10.5:
        raise ValueError('At least two steps and a memory limit <=10.5 GiB are required')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / 'benchmark.json'
    if profile_path.exists():
        raise FileExistsError(f'Benchmark already exists: {profile_path}')
    config = resolve_config(config_path, data_root, output)
    comparison = assert_paper_config(config)
    selected_device = torch.device(device)
    if selected_device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('The full paper memory benchmark requires CUDA')
    selected_device = indexed_cuda_device(selected_device, current_index=torch.cuda.current_device())
    total_gib = torch.cuda.get_device_properties(selected_device).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1., max_reserved_gib / total_gib), selected_device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(selected_device)
    profile = {'complete': False, 'passed': False, 'diagnostic': True,
               'started_utc': utc_now(), 'device': str(selected_device),
               'batch_size': config['batch_size'], 'eval_batch_size': config['eval_batch_size'],
               'amp_dtype': 'float16', 'grad_scaler': bool(config['amp']),
               'original_paper_recipe': comparison['matches'], 'config_sha256': sha256(config_path),
               'source_sha256': source_hashes(), 'max_reserved_gib': max_reserved_gib,
               'device_preflight': {'free_gib': free_bytes / 1024**3, 'total_gib': total_bytes / 1024**3,
                                    'allocator_fraction': min(1., max_reserved_gib / total_gib)},
               'runtime_overrides': {'pin_memory': False} if disable_host_pinning else {},
               'runtime_override_reason': 'WSL pinned-host ceiling measured: one768MiB allocation passed; cumulative1729MiB failed before forward.'
                                          if disable_host_pinning else None,
               'environment': environment_record(selected_device),
               'note': 'Longest actual pair stress test; repeated batches are not paper training. Gradient observation synchronizes CUDA.'}
    atomic_json(output / 'benchmark_status.json', profile)
    try:
        validation_start = time.monotonic()
        validation = validate_data(config, splits=('train', 'val'), inspect_features=True)
        atomic_json(output / 'input_validation.json', validation)
        if not validation['valid']:
            raise ValueError('Input validation failed: ' + '; '.join(validation['errors']))
        profile['validation_seconds'] = time.monotonic() - validation_start
        profile['input_sha256'] = input_hashes(config)
        trainer = load_trainer()
        trainer.set_seed(int(config['seed']))
        train_original = trainer.make_loader(config, 'train', config['graph_version'], config['batch_size'], False)
        val_original = trainer.make_loader(config, 'val', config['graph_version'], config['eval_batch_size'], False)
        if any(len(loader.dataset) != validation['splits'][split]['rows']
               for loader, split in ((train_original, 'train'), (val_original, 'val'))):
            raise ValueError('Legacy loader filtered the benchmark inputs')

        def selected_loader(loader, indices, repeats):
            kwargs = profile_loader_options(loader, disable_host_pinning=disable_host_pinning)
            return DataLoader(loader.dataset, batch_sampler=[indices] * repeats, **kwargs)

        train_indices = longest_pair_indices(train_original.dataset.rows, validation, config['batch_size'])
        val_indices = longest_pair_indices(val_original.dataset.rows, validation, config['eval_batch_size'])
        train_loader = selected_loader(train_original, train_indices, steps)
        val_loader = selected_loader(val_original, val_indices, 1)
        profile['selected_pairs'] = {}
        for split, loader, indices in (('train', train_original, train_indices), ('val', val_original, val_indices)):
            rows = loader.dataset.rows.iloc[indices]
            profile['selected_pairs'][split] = [
                {'row_index': index, 'pair_id': str(row.pair_id), 'sequence_hash': str(row.sequence_hash),
                 'nodes': validation['features']['graphs'][str(row.sequence_hash)]['nodes']}
                for index, row in zip(indices, rows.itertuples())]
        profile['longest_pairs_verified'] = True
        model = trainer.DTIModel('E4_GAUSS_NODE_FUSION', config).to(selected_device)
        loss_fn = trainer.make_loss(config).to(selected_device)
        contrastive = trainer.make_contrastive_context(config, selected_device)
        optimizer = trainer.make_optimizer(config, model)
        hypergradient = trainer.make_hypergradient(config, optimizer)
        scaler = torch.amp.GradScaler(enabled=True)
        named = list(model.named_parameters())
        before = {name: parameter.detach().cpu().clone() for name, parameter in named if parameter.requires_grad}
        observations = []

        def observe_step(opt, args, kwargs):
            gradients = [(name, param.grad.detach()) for name, param in named if param.grad is not None]
            finite = all(bool(torch.isfinite(grad).all()) for _, grad in gradients)
            norms = gradient_group_totals(gradients)
            observations.append({'step': len(observations) + 1, 'finite_gradients': finite,
                                 'group_gradient_l1': norms, 'lr': opt.param_groups[0]['lr']})

        hook = optimizer.register_step_post_hook(observe_step)
        torch.cuda.synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
        train_start = time.monotonic()
        train_loss, auxiliary = trainer.train_one_epoch(model, train_loader, optimizer, scaler,
            loss_fn, selected_device, config, contrastive_ctx=contrastive, hypergradient=hypergradient)
        torch.cuda.synchronize(selected_device)
        profile['train_seconds'] = time.monotonic() - train_start
        profile['train_peak_reserved_gib'] = torch.cuda.max_memory_reserved(selected_device) / 1024**3
        hook.remove()
        val_start = time.monotonic()
        val_metrics = trainer.evaluate(model, val_loader, loss_fn, selected_device, amp=True,
                                       save_predictions=output / 'validation_sample_predictions.csv')
        torch.cuda.synchronize(selected_device)
        profile['validation_sample_seconds'] = time.monotonic() - val_start
        changed = [name for name, param in named if name in before and
                   not torch.equal(before[name], param.detach().cpu())]
        profile.update(complete=True, optimizer_steps=len(observations), requested_train_steps=steps,
            skipped_scaler_steps=steps-len(observations), gradient_observations=observations,
            parameters_changed=changed, parameter_update_count=len(changed), train_loss=float(train_loss),
            auxiliary=auxiliary, validation_sample_metrics=val_metrics,
            finite_losses=math.isfinite(train_loss) and math.isfinite(val_metrics['Loss']),
            finite_gradients=bool(observations) and all(item['finite_gradients'] for item in observations),
            gradient_groups_nonzero={group: any(item['group_gradient_l1'][group] > 0 for item in observations)
                                     for group in ('dynamic', 'alpha', 'variance')},
            contrastive_enabled=contrastive is not None, auxiliary_enabled=config['unimodal_aux']['enabled'],
            hypergradient_enabled=hypergradient is not None,
            hypergradient_updated=math.isfinite(auxiliary.get('hypergrad_cos', math.nan)),
            scaler_final_scale=float(scaler.get_scale()),
            peak_reserved_gib=torch.cuda.max_memory_reserved(selected_device) / 1024**3,
            peak_allocated_gib=torch.cuda.max_memory_allocated(selected_device) / 1024**3,
            finished_utc=utc_now(), passed=True)
        if not changed:
            profile['passed'] = False
        assert_benchmark_ready(profile)
        atomic_json(profile_path, _json_finite(profile))
        atomic_json(output / 'benchmark_status.json', _json_finite(profile))
        return profile
    except BaseException as error:
        profile.update(complete=False, passed=False, error=str(error), error_type=type(error).__name__,
                       failed_utc=utc_now())
        atomic_json(profile_path, _json_finite(profile))
        atomic_json(output / 'benchmark_status.json', _json_finite(profile))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--max-reserved-gib', type=float, default=10.5)
    parser.add_argument('--disable-host-pinning', action='store_true',
                        help='Explicit local host-memory resource override; computation and batch sizes unchanged')
    args = parser.parse_args(argv)
    record = benchmark(args.config, args.data_root, args.output, device=args.device, steps=args.steps,
                       max_reserved_gib=args.max_reserved_gib, disable_host_pinning=args.disable_host_pinning)
    print(json.dumps({'passed': record['passed'], 'peak_reserved_gib': record['peak_reserved_gib'],
                      'optimizer_steps': record['optimizer_steps']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
