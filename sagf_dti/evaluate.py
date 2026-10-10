"""Reload a selected checkpoint and evaluate its stated split."""
from __future__ import annotations

from pathlib import Path

from .artifacts import atomic_json, recipe_identity, sha256, utc_now


def evaluate_checkpoint(config_path, data_root, checkpoint, output, *, split='test', device=None,
                        disable_host_pinning=False):
    import torch
    from .config import resolve_config
    from .data import validate_data
    from .legacy import load_trainer
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    checkpoint = Path(checkpoint).expanduser().resolve()
    config = resolve_config(config_path, data_root)
    selected_device = torch.device(device or config.get('device', 'cpu'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('experiment') != 'E4_GAUSS_NODE_FUSION':
        raise ValueError('Checkpoint is not from the SAGF-DTI Gaussian experiment')
    if recipe_identity(state['config']) != recipe_identity(config):
        raise ValueError('Checkpoint configuration does not match the requested recipe/seed')
    check = validate_data(config, splits=(split,), inspect_features=True)
    if not check['valid']:
        raise ValueError('Input validation failed: ' + '; '.join(check['errors']))
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Evaluation output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    trainer = load_trainer()
    model = trainer.DTIModel('E4_GAUSS_NODE_FUSION', config).to(selected_device)
    model.load_state_dict(state['model'], strict=True)
    runtime_overrides = {'pin_memory': False} if disable_host_pinning else {}
    loader = trainer.make_loader({**config, **runtime_overrides}, split, config['graph_version'],
                                config['eval_batch_size'], False)
    if len(loader.dataset) != check['splits'][split]['rows']:
        raise ValueError('Legacy loader filtered evaluation membership')
    loss = trainer.make_loss(config).to(selected_device)
    predictions = output / f'{split}_predictions.csv'
    metrics = trainer.evaluate(model, loader, loss, selected_device,
                               amp=config.get('amp', True), save_predictions=predictions)
    result = {'complete': True, 'split': split, 'checkpoint_sha256': sha256(checkpoint),
              'checkpoint_epoch': state['epoch'], 'metrics': metrics,
              'n_pairs': len(loader.dataset), 'predictions_sha256': sha256(predictions),
              'runtime_overrides': runtime_overrides,
              'completed_utc': utc_now()}
    atomic_json(output / 'evaluation.json', result)
    return result
