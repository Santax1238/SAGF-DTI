from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
import yaml

from sagf_dti.train import train
from sagf_dti.evaluate import evaluate_checkpoint


@pytest.fixture
def toy_paper_inputs(tmp_path):
    graph_dir = tmp_path / 'graphs/esmdance_node_cond35_dyn50'
    graph_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(71)
    for index in range(4):
        target = f'p{index}'
        n = index + 2
        esm = torch.randn(n, 3, generator=generator)
        dynamic = torch.rand(n, 50, generator=generator)
        graph = Data(x=esm, x_esm=esm, x_dyn=dynamic,
                     predicted_dynamic_residue_features=dynamic,
                     esmdance_35m_embedding=torch.randn(n, 5, generator=generator),
                     esmdance_embedding_mask=torch.ones(n, 1),
                     edge_index=torch.empty(2, 0, dtype=torch.long),
                     target_id=target, sequence_hash=target,
                     graph_version='esmdance_node_cond35_dyn50', effective_len=n)
        torch.save(graph, graph_dir / f'{target}.pt')
    for split, n in (('train', 8), ('val', 4), ('test', 4)):
        pd.DataFrame({'pair_id': [f'{split}{i}' for i in range(n)],
                      'drug_id': [f'd{i%2}' for i in range(n)],
                      'sequence_hash': [f'p{i%4}' for i in range(n)],
                      'label': [i%2 for i in range(n)]}).to_csv(tmp_path / f'{split}.csv', index=False)
    np.savez(tmp_path / 'kpgt.npz', drug_ids=np.array(['d0', 'd1'], dtype=object),
             fps=np.array([[1,2,3,4], [4,3,2,1]], dtype=np.float32))
    cfg = {'experiment_id':'E4_GAUSS_NODE_FUSION', 'graph_version':'esmdance_node_cond35_dyn50',
           'run_name':'toy_seed42', 'seed':42, 'device':'cpu', 'dataset':'SNAP', 'biofusion_setting':'prot',
           'batch_size':4, 'eval_batch_size':4, 'epochs':2, 'patience':1, 'amp':False,
           'num_workers':0, 'pin_memory':False, 'lr':.0001, 'weight_decay':.0001, 'loss':'bce', 'save_best_by':'AUROC',
           'model':{'drug_dim':4, 'esm_dim':3, 'dyn_dim':50, 'esmdance35_dim':5, 'hidden_dim':8,
                    'edge_dim':36, 'output_dim':2, 'dropout':.1, 'dyn_conditioned_on_esm35':True,
                    'gaussian_tempered_poe':True, 'gaussian_semantic_anchor':True,
                    'pooling_type':'drug_conditioned_attention'},
           'unimodal_aux':{'enabled':True,'lambda_uni':.2,'lambda_rel':.03,'temperature':1},
           'hypergradient':{'enabled':True,'mode':'cauchy_lr','beta':.01,'min_lr':1e-6,'max_lr':.0002},
           'contrastive':{'enabled':False},
           'data':{f'{split}_csv':str(tmp_path / f'{split}.csv') for split in ('train','val','test')}}
    cfg['data'].update(graph_root=[str(tmp_path / 'graphs')],
                       bindingdb_drug_npz=[str(tmp_path / 'kpgt.npz')],
                       davis_drug_npz=[str(tmp_path / 'kpgt.npz')])
    config = tmp_path / 'toy.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return tmp_path, config


def test_training_rejects_nonfinite_auxiliary_before_creating_run(toy_paper_inputs):
    root, config = toy_paper_inputs
    path = root / 'graphs/esmdance_node_cond35_dyn50/p0.pt'
    graph = torch.load(path, weights_only=False)
    graph.esmdance_35m_embedding[0, 0] = float('nan')
    torch.save(graph, path)
    with pytest.raises(ValueError, match='finite'):
        train(config, root, root / 'runs', device='cpu', dry_run=True, max_batches=1)
    assert not (root / 'runs/toy_seed42').exists()


def test_fresh_training_locks_then_reloads_selected_checkpoint(toy_paper_inputs):
    root, config = toy_paper_inputs
    result = train(config, root, root / 'runs', device='cpu')
    assert result['complete'] and not result['diagnostic']
    assert result['original_paper_recipe'] is False
    assert json.loads((root / 'runs/toy_seed42/training_manifest.json').read_text())['runtime_overrides'] == {}
    folder = root / 'runs/toy_seed42'
    lock = json.loads((folder / 'checkpoint_lock.json').read_text())
    assert lock['test_used_for_selection'] is False
    reloaded = evaluate_checkpoint(config, root, folder / 'checkpoints/best.pt', root / 'reload', device='cpu')
    assert reloaded['n_pairs'] == 4
    for metric in ('AUROC','AUPR','ACC','Loss'):
        assert reloaded['metrics'][metric] == pytest.approx(result['test_metrics'][metric], abs=1e-12)
    assert reloaded['predictions_sha256'] == result['verified_files']['test_predictions.csv']
    with pytest.raises(FileExistsError):
        train(config, root, root / 'runs', device='cpu')
    changed = yaml.safe_load(config.read_text())
    changed['seed'] = 1234
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='recipe/seed'):
        evaluate_checkpoint(config, root, folder / 'checkpoints/best.pt', root / 'wrong', device='cpu')


def test_explicit_host_pinning_override_preserves_recorded_recipe(toy_paper_inputs, monkeypatch):
    from sagf_dti.legacy import load_trainer
    root, config = toy_paper_inputs
    cfg = yaml.safe_load(config.read_text())
    cfg['pin_memory'] = True
    config.write_text(yaml.safe_dump(cfg))
    trainer = load_trainer()
    original_loader = trainer.make_loader
    observed = []

    def observe_loader(cfg, *args, **kwargs):
        observed.append((cfg['pin_memory'], cfg['num_workers'], cfg['seed']))
        return original_loader(cfg, *args, **kwargs)

    monkeypatch.setattr(trainer, 'make_loader', observe_loader)
    result = train(config, root, root / 'runs', device='cpu', disable_host_pinning=True)
    folder = root / 'runs/toy_seed42'
    state = torch.load(folder / 'checkpoints/best.pt', weights_only=False)
    assert state['config']['pin_memory'] is True
    manifest = json.loads((folder / 'training_manifest.json').read_text())
    assert manifest['runtime_overrides'] == {'pin_memory': False}
    assert result['runtime_overrides'] == {'pin_memory': False}
    reloaded = evaluate_checkpoint(config, root, folder / 'checkpoints/best.pt', root / 'reload',
                                   device='cpu', disable_host_pinning=True)
    assert reloaded['runtime_overrides'] == {'pin_memory': False}
    assert observed == [(False, 0, 42)] * 4
    assert reloaded['predictions_sha256'] == result['verified_files']['test_predictions.csv']
