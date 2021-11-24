from pathlib import Path
import torch


def save_snapshot(agent, step, root):
    snapshot_dir = root / Path('snapshots')
    snapshot_dir.mkdir(exist_ok=True, parents=True)
    snapshot = snapshot_dir / f'snapshot_{step}.pt'
    payload = dict(agent=agent, step=step)
    with snapshot.open('wb') as f:
        torch.save(payload, f)

def load_snapshot(snapshot):
    snapshot = Path(snapshot)
    if not snapshot.exists():
        raise ValueError(f'snapshot {str(snapshot)} does not exist')

    with snapshot.open('rb') as f:
        agent = torch.load(f)['agent']
        return agent
