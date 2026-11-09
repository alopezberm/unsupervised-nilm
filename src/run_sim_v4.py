"""Score v4 on the tuned simulator (sim3) to see how far the real score is from the model's floor."""
import pickle
import time
import numpy as np
import data
import disaggregate as d
import simulate as S

if __name__ == '__main__':
    X, temp, ts = data.load()
    sim = pickle.load(open('../data/sim3.pkl', 'rb'))
    t0 = time.time()
    r = d.run('v4', data=(sim['Xs'], temp, ts), kernel=sim['kernel'])
    sc, per = S.score(r.pred, sim['truth'])
    print(f'v4 on sim3: {sc:.5f}  ' + '  '.join(f'{c} {v:.5f}' for c, v in per.items()) + f'   ({time.time() - t0:.0f}s)', flush=True)
