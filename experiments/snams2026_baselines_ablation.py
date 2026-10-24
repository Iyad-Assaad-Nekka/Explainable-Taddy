# =============================================================================
#  SNAMS 2026 camera-ready -- extra experiments for the X-TADDY paper
#  (reviewer requests: baselines on all 3 metrics, ablation, statistical
#   tests, one worked example, computational cost)
#  Same protocol as the accepted paper: repo run_evaluation.py setup (TADDY
#  patches, 0_prepare_data, 50/50 split, 10% anomalies, seed 1, 100 epochs),
#  top-50 true-positive edges per dataset, top-k = 5, D = 14, Ns = 150.
#  All metrics use the paper's embedding-space evaluation (Sec. IV-E):
#    Fid = 1 - |s* - s_topk| ; Suff = s_topk ; Comp = s* - s_without_topk
#  CPU is enough. Runtime ~1.5-2 h (mostly TADDY training).
# =============================================================================
import os, sys, time, json, subprocess, shutil
if not os.path.isdir('/content/xt_repo'):
    subprocess.run('git clone -q --depth 1 https://github.com/Iyad-Assaad-Nekka/Explainable-Taddy.git /content/xt_repo', shell=True, check=True)
if not os.path.isdir('/content/xtaddy'): shutil.copytree('/content/xt_repo/xtaddy', '/content/xtaddy')
src = open('/content/xt_repo/examples/run_evaluation.py').read()
exec(src[:src.index('# ── 4.')])                      # official setup: clone + patch TADDY, prepare data
import numpy as np, torch, scipy.special, scipy.stats
from sklearn import metrics
from sklearn.linear_model import Ridge, LinearRegression
from xtaddy import RandomExplainer, AttentionExplainer, GradientExplainer
TOPK, NS, NEDGES = 5, 150, int(os.environ.get('CR_NEDGES', 50))
EPOCHS = int(os.environ.get('CR_EPOCHS', 100)); DATASETS = os.environ.get('CR_DATASETS', 'uci,btc_alpha,btc_otc').split(',')
OUT = '/content/snams_cr'; os.makedirs(OUT, exist_ok=True)

# ---------------- masking / scoring ----------------
def emb_of(model, ii, hh, tt):
    with torch.no_grad(): return model.transformer.embeddings(init_pos_ids=ii, hop_dis_ids=hh, time_dis_ids=tt).detach()
def score_emb(model, h_base, mask):                 # paper's embedding-space masking
    h = h_base.clone(); a = mask.bool()
    h[0, ~a, :] = h[0, a, :].mean(0) if a.sum() > 0 else 0.0
    with torch.no_grad(): out, _ = model.forward_from_embeddings(h)
    return torch.sigmoid(out).item()
def score_idx(model, ii, hh, tt, mask):             # ablation: naive index-space masking (ids -> 0)
    a = mask.bool(); i2, h2, t2 = ii.clone(), hh.clone(), tt.clone()
    i2[0, ~a] = 0; h2[0, ~a] = 0; t2[0, ~a] = 0
    with torch.no_grad(): out, _ = model.forward(i2, h2, t2)
    return torch.sigmoid(out).item()
def kernel_w(Z, D):
    w = []
    for s in Z.sum(1).astype(int):
        if s == 0 or s == D: w.append(1000.0)
        else: w.append(float((D - 1) / (scipy.special.comb(D, s) * s * (D - s))))
    return np.clip(np.array(w), 0, 1000)
def coalitions(D, rng, smart=True):
    Z = np.zeros((NS, D), np.float32)
    if smart:                                        # paper's Smarter sampler
        Z[0] = 1; i = 2
        for j in range(D):
            if i >= NS: break
            Z[i] = 1; Z[i, j] = 0; i += 1
            if i >= NS: break
            Z[i, j] = 1; i += 1
        Z[i:] = rng.integers(0, 2, size=(NS - i, D))
    else:                                            # ablation: plain uniform random coalitions
        Z[:] = rng.integers(0, 2, size=(NS, D))
    return Z
def shapley(model, ii, hh, tt, h_base, rng, smart=True, solver='wls', masking='emb'):
    D = h_base.shape[1]; Z = coalitions(D, rng, smart)
    f = np.array([score_emb(model, h_base, torch.from_numpy(z)) if masking == 'emb'
                  else score_idx(model, ii, hh, tt, torch.from_numpy(z)) for z in Z])
    if solver == 'wls': reg = Ridge(alpha=0.01).fit(Z, f, sample_weight=kernel_w(Z, D))
    else:               reg = LinearRegression().fit(Z, f)          # ablation: ordinary least squares
    return reg.coef_
def evaluate(model, h_base, s_true, phi):
    D = h_base.shape[1]; top = np.argsort(np.abs(phi))[-TOPK:]
    m = torch.zeros(D).bool(); m[top] = True
    s_top = score_emb(model, h_base, m); s_rest = score_emb(model, h_base, ~m)
    return dict(fid=1 - abs(s_true - s_top), suff=s_top, comp=s_true - s_rest)

def grad_x_input(model, h_base):
    e = h_base.clone().requires_grad_(True)
    enc, _ = model.transformer.encoder(e)
    seq = sum(enc[:, i, :] for i in range(model.config.k + 1)) / float(model.config.k + 1)
    torch.sigmoid(model.cls_y(seq)).sum().backward()
    return (e.grad[0] * e.detach()[0]).sum(-1).numpy()

METHODS = ['Random', 'Attention', 'Gradient', 'GradxInput', 'X-TADDY',
           'Abl: index-space masking', 'Abl: random sampler', 'Abl: OLS (no kernel weights)']
ALL, EXAMPLE, TIMING = {}, {}, {}
for dataset in DATASETS:
    print(f'\n===== {dataset} =====', flush=True)
    np.random.seed(1); torch.manual_seed(1)          # paper's seed
    data_obj = DynamicDatasetLoader(); data_obj.dataset_name = dataset; data_obj.k = 5; data_obj.window_size = 2
    data_obj.anomaly_per = 0.1; data_obj.train_per = 0.5; data_obj.load_all_tag = False; data_obj.compute_s = True
    config = MyConfig(k=5, window_size=2, hidden_size=32, intermediate_size=32, num_attention_heads=2,
                      num_hidden_layers=2, weight_decay=5e-4)
    class Args:
        anomaly_per = 0.1; train_per = 0.5; neighbor_num = 5; window_size = 2; embedding_dim = 32
        num_hidden_layers = 2; num_attention_heads = 2; lr = 0.001; weight_decay = 5e-4; seed = 1; print_feq = 10
    Args.dataset = dataset; Args.max_epoch = 100
    model = DynADModel(config, Args()); model.lr = 0.001; model.max_epoch = 100
    Settings().prepare(data_obj, model); ld = data_obj.load(); model.data = ld
    model.train_model(EPOCHS)
    _, _, hop_emb, int_emb, time_emb = model.generate_embedding(ld['edges'])
    model.eval(); preds, ys = [], []
    t0 = time.time()
    for snap in ld['snap_test']:
        with torch.no_grad(): p, _ = model.forward(int_emb[snap], hop_emb[snap], time_emb[snap])
        preds.append(torch.sigmoid(p).squeeze().numpy()); ys.append(ld['y'][snap].numpy())
    n_test = sum(len(p) for p in preds); t_infer = (time.time() - t0) / n_test
    auc = metrics.roc_auc_score(np.hstack(ys), np.hstack(preds)); print(f'detection AUC {auc:.4f}', flush=True)
    cand = sorted([(float(preds[si][ei]), snap, ei) for si, snap in enumerate(ld['snap_test'])
                   for ei in range(len(preds[si])) if ys[si][ei] == 1], reverse=True)[:NEDGES]
    base = {}                                        # repo baselines (their phi), our 3 metrics
    for name, cls in [('Random', RandomExplainer), ('Attention', AttentionExplainer), ('Gradient', GradientExplainer)]:
        base[name] = [r.phi for r in cls(model, top_k=TOPK).explain_edges(int_emb, hop_emb, time_emb, cand)]
    res = {m: [] for m in METHODS}; tm = {m: 0.0 for m in METHODS}
    rng = {m: np.random.default_rng(42) for m in METHODS}
    for k, (s, snap, ei) in enumerate(cand):
        ii, hh, tt = int_emb[snap][ei:ei+1], hop_emb[snap][ei:ei+1], time_emb[snap][ei:ei+1]
        h = emb_of(model, ii, hh, tt)
        with torch.no_grad(): s_true = torch.sigmoid(model.forward(ii, hh, tt)[0]).item()
        phis = {}
        for name in ['Random', 'Attention', 'Gradient']: phis[name] = base[name][k]
        a = time.time(); phis['GradxInput'] = grad_x_input(model, h); tm['GradxInput'] += time.time() - a
        for name, kw in [('X-TADDY', {}), ('Abl: index-space masking', dict(masking='idx')),
                         ('Abl: random sampler', dict(smart=False)), ('Abl: OLS (no kernel weights)', dict(solver='ols'))]:
            a = time.time(); phis[name] = shapley(model, ii, hh, tt, h, rng[name], **kw); tm[name] += time.time() - a
        for name in METHODS: res[name].append(evaluate(model, h, s_true, phis[name]))
        if k == 0:                                   # worked example: top-scored true anomaly
            phi = phis['X-TADDY']; W = model.config.window_size; T = model.config.k + 2
            EXAMPLE[dataset] = dict(score=s_true, snap=int(snap), edge_index=int(ei),
                                    tokens=[[int(x) for x in ii[0]], [int(x) for x in hh[0]], [int(x) for x in tt[0]]],
                                    phi=phi.tolist(), spatial=[float(phi[l*T + i] + 0) for l in range(W) for i in range(T)],
                                    phi_spatial=[float(sum(phi[l*T + i] for l in range(W))) for i in range(T)],
                                    phi_temporal=[float(phi[l*T:(l+1)*T].sum()) for l in range(W)])
        if (k + 1) % 10 == 0: print(f'  {k+1}/{len(cand)} edges', flush=True)
    ALL[dataset] = dict(auc=auc, per_edge=res)
    TIMING[dataset] = dict(taddy_infer_ms_per_edge=1000 * t_infer, **{m + '_s_per_edge': tm[m] / len(cand) for m in tm})
    json.dump(dict(ALL=ALL, EXAMPLE=EXAMPLE, TIMING=TIMING), open(f'{OUT}/results.json', 'w'))

# ---------------- tables ----------------
def ci(x, B=2000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x)
    b = [r.choice(x, len(x)).mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
print('\n================ TABLE A: baselines + ablation (top-50 TP, k=5) ================')
for metric in ['fid', 'suff', 'comp']:
    print(f'\n--- {metric.upper()} (mean [95% bootstrap CI]) ---')
    print(f"{'Method':32s}" + ''.join(f'{d:>26s}' for d in ALL) + f"{'Avg':>10s}")
    for m in METHODS:
        row, avg = '', []
        for d in ALL:
            v = [e[metric] for e in ALL[d]['per_edge'][m]]; lo, hi = ci(v); avg.append(np.mean(v))
            row += f'{np.mean(v):9.4f} [{lo:.4f},{hi:.4f}]'
        print(f'{m:32s}{row}{np.mean(avg):10.4f}')
print('\n================ TABLE B: paired Wilcoxon, X-TADDY vs each method (pooled 150 edges) ================')
for metric in ['fid', 'comp']:
    xt = np.concatenate([[e[metric] for e in ALL[d]['per_edge']['X-TADDY']] for d in ALL])
    for m in METHODS:
        if m == 'X-TADDY': continue
        o = np.concatenate([[e[metric] for e in ALL[d]['per_edge'][m]] for d in ALL])
        p = scipy.stats.wilcoxon(xt, o, zero_method='zsplit').pvalue if np.any(xt != o) else 1.0
        print(f'{metric:5s} X-TADDY {xt.mean():.4f} vs {m:30s} {o.mean():.4f}  diff {xt.mean()-o.mean():+.4f}  p={p:.2e}')
print('\n================ TABLE C: cost ================')
for d, t in TIMING.items(): print(d, {k: round(v, 4) for k, v in t.items()})
print('\n================ WORKED EXAMPLE (top TP per dataset) ================')
for d, e in EXAMPLE.items():
    print(d, 'score', round(e['score'], 4), '| phi_spatial [u, v, n1..n5]:', np.round(e['phi_spatial'], 4).tolist(),
          '| phi_temporal [l=0, l=1]:', np.round(e['phi_temporal'], 4).tolist())
print(f'\nsaved {OUT}/results.json  -> download it and send it to me')
