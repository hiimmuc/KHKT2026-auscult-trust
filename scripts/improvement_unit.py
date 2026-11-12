"""One resumable final fit followed by its frozen exports; no target-based selection."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import runs
from src.artifacts import atomic_json, file_hashes, sha256_file, verify_file
from src.training.config import parse_args


def checked_step(command, outputs, checkpoint_sha, receipt):
    expected = {'command': command, 'checkpoint_sha256': checkpoint_sha}
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if any(saved[k] != v for k, v in expected.items()):
            raise ValueError('Changed frozen inference step: ' + str(receipt))
        for path, digest in saved['files'].items():
            verify_file(path, digest)
        return
    subprocess.run(command, check=True)
    atomic_json(receipt, {**expected, 'files': file_hashes(outputs)})


def main():
    args = parse_args()
    if args.selection != 'fixed' or args.cv_folds or len(args.report_epoch_list) != 1:
        raise ValueError('Final pipeline needs fixed selection and one source-selected epoch')
    out = runs.unit_dir(args.exp, args.cell, args.unit, 'outputs')
    checkpoint = runs.unit_dir(args.exp, args.cell, args.unit, 'checkpoints') / f'report_epoch_{args.report_epoch_list[0]}.pth'
    marker = out / 'pipeline-complete.json'
    if Path(os.environ['PIPELINE_DONE']).resolve() != marker.resolve():
        raise ValueError('Pipeline marker does not match this training unit')
    subprocess.run([sys.executable, '-u', '-m', 'src.training.main', *sys.argv[1:]], check=True)
    checkpoint_sha = sha256_file(checkpoint)
    outputs = [out / name for name in ('train_args.json', 'report.json', 'test_preds.npz', 'sc_state.npz')]
    if args.holdout_device:
        outputs.append(out / 'cal_preds.npz')
    else:
        cycle = out / 'export/test.npz'
        checked_step([sys.executable, '-m', 'src.training.export_probs', str(out), '--ckpt', checkpoint.name,
                      '--num_workers', '0'], [cycle, out / 'export/model.sha256'], checkpoint_sha, out / 'cycle-export.complete.json')
        outputs += [cycle, out / 'export/model.sha256']
        for name, command in [('screening', ['src.pipelines.screening_infer']), ('kauh', ['src.pipelines.kauh_infer'])]:
            prediction = out / (name + '.npz')
            checked_step([sys.executable, '-m', *command, str(out), '--ckpt', checkpoint.name,
                          '--out', str(prediction), '--sc'], [prediction], checkpoint_sha, out / (name + '.complete.json'))
            outputs.append(prediction)
    atomic_json(marker, {'checkpoint_sha256': checkpoint_sha, 'report_epoch': args.report_epoch_list[0],
                        'protocol': 'lodo-train-reference-v2', 'files': file_hashes(outputs)})


if __name__ == '__main__':
    main()
