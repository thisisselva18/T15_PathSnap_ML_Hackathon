"""Metrics: row-level (acc, P, R, F1, macro F1, log loss), observation Top-1, candidate miss rate, route continuity, sequence-level.

Owner: P2
"""

def top1_accuracy(df, score_col="score", observations=None):
    """Observation Top-1; optional manifest includes observations with no candidates as wrong."""
    import numpy as np
    if df.empty: return float('nan')
    scored=df.sort_values(score_col,ascending=False).drop_duplicates(['trip_id','obs_idx'])
    if observations is None: return float(np.mean(scored.arc_id.to_numpy()==scored.ref_arc_id.to_numpy()))
    predicted={(r.trip_id,int(r.obs_idx)):int(r.arc_id==r.ref_arc_id) for r in scored.itertuples()}
    return float(np.mean([predicted.get((r.trip_id,int(r.obs_idx)),0) for r in observations.itertuples()]))


def candidate_miss_rate(df, observations=None):
    """Fraction of observations whose candidate set omits its reference arc."""
    if df.empty: return float('nan')
    missed=0; total=0
    for _,g in df.groupby(['trip_id','obs_idx']):
        total+=1; missed+=int(int(g.ref_arc_id.iloc[0]) not in set(g.arc_id.astype(int)))
    if observations is None: return missed/total if total else float('nan')
    observed={(tid,int(obs)):set(g.arc_id.astype(int)) for (tid,obs),g in df.groupby(['trip_id','obs_idx'])}
    return float(np.mean([int(int(r.ref_arc_id) not in observed.get((r.trip_id,int(r.obs_idx)),set())) for r in observations.itertuples()])) if len(observations) else float('nan')


def route_continuity(pred_arcs, arcs):
    """Fraction of consecutive predicted arcs whose directed endpoints connect."""
    import numpy as np
    pred=np.asarray(pred_arcs,int); arcs=np.asarray(arcs,int)
    if len(pred)<2: return 1.0
    ok=[arcs[a,1]==arcs[b,0] for a,b in zip(pred[:-1],pred[1:]) if a<len(arcs) and b<len(arcs)]
    return float(np.mean(ok)) if ok else float('nan')
