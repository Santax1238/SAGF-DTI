#!/usr/bin/env python3
"""Wait for current validation, extract all blocks, then publish feature patches."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from window_extractor import HERE, now, sha, write


def main():
    lock = (HERE/'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = HERE/'pipeline_status.json'
    status = {'status': 'waiting_for_in_progress_source_validation', 'pid': os.getpid(), 'started_utc': now(),
              'window_wrapper_sha256': sha(HERE/'window_extractor.py'), 'aggregation_sha256': sha(HERE/'aggregate_windows.py'),
              'workers': 8, 'model_or_test_inference_performed': False}
    write(status_path, status)
    try:
        deadline = time.monotonic() + 7200
        while not (HERE/'validation_receipt.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Validation has not completed within2h; no extraction started')
            if 'Traceback (most recent call last)' in (HERE/'validation.log').read_text():
                raise ValueError('Source validation failed; read validation.log')
            time.sleep(15)
        validation = json.loads((HERE/'validation_receipt.json').read_text())
        assert validation['passed'] and validation['wrapper_sha256'] == status['window_wrapper_sha256']
        for name, key in [('window_extractor.py', 'window_wrapper_sha256'), ('aggregate_windows.py', 'aggregation_sha256')]:
            assert sha(HERE/name) == status[key]
        status.update(status='extracting_all775_trajectory_mappings_two_windows', utc=now())
        write(status_path, status)
        subprocess.run([sys.executable, '-B', str(HERE/'window_extractor.py'), 'run', '--workers', '8'], check=True)
        progress = json.loads((HERE/'progress.json').read_text())
        assert progress['status'] == 'all_window_features_extracted_pending_target_aggregation'
        status.update(status='aggregating_feature_patches', utc=now())
        write(status_path, status)
        subprocess.run([sys.executable, '-B', str(HERE/'aggregate_windows.py')], check=True)
        status.update(status='trajectory_window_patches_complete', finished_utc=now(),
                      patch_manifest=str(HERE/'patches/PATCH_MANIFEST.json'),
                      patch_manifest_sha256=sha(HERE/'patches/PATCH_MANIFEST.json'))
        write(status_path, status)
    except BaseException as exc:
        status.update(status='failed_requires_diagnosis', error=repr(exc), utc=now())
        write(status_path, status)
        raise


if __name__ == '__main__':
    main()
