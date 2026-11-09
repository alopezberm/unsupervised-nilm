"""Fit any version (argument, e.g. v4 or v5) on the real data and write the next numbered submission (parallel HMM needs a main guard)."""
import sys
import time
import numpy as np
import disaggregate as d
import submission as s

if __name__ == '__main__':
    t0 = time.time()
    version = sys.argv[1] if len(sys.argv) > 1 else 'v3'
    r = d.run(version)
    np.savez_compressed(f'../data/base_{version}.npz', **r.pred)
    path, rep = s.save(r.pred, r.ts, version, total=r.X)
    print(f'REAL {version} done in {time.time() - t0:.0f}s -> {path}  ready={rep["READY TO SUBMIT"]}', flush=True)
    for c in r.pred:
        print(f'  mean {c}: {r.pred[c].mean():.4f}', flush=True)
