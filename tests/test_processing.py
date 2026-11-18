import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.features import FEATURES, add_features
from pathsnap.geo import point_segment
from pathsnap.labels import align_points_to_route
from pathsnap.segment import split_segments
from pathsnap.io import clean_track
from scripts.train_models import top1, row_metrics

def test_candidate_generation_returns_arc_ids_and_metric_distances():
    A=np.array([[0.,0.],[0.,10.]])
    B=np.array([[10.,0.],[10.,10.]])
    c=generate_candidates(np.array([[5.,1.]]),A,B,np.array([0,1]),radius_m=5,k=2)
    assert c.arc_id.tolist()==[0]
    assert np.isclose(c.dist.iloc[0],1.)

def test_route_alignment_is_monotone_and_uses_arc_row_ids():
    A=np.array([[0.,0.],[10.,0.]])
    B=np.array([[10.,0.],[20.,0.]])
    pts=np.array([[2.,1.],[12.,1.]])
    labels,dist=align_points_to_route(pts,np.array([0.,1.]),np.array([0,0]),np.array([0,1]),A,B)
    assert labels.tolist()==[0,1]
    assert np.allclose(dist,[1.,1.])

def test_feature_builder_is_causal_and_schema_aligned():
    A=np.array([[0.,0.],[0.,10.]])
    B=np.array([[10.,0.],[10.,10.]])
    c=pd.DataFrame({'obs_idx':[0], 'arc_id':[0], 'dist':[1.], 'cand_rank':[0], 't_raw':[.5], 't_frac':[.5]})
    args=(np.array([0,0,0]),A,B,np.array([0,0]),np.array([0,0]),np.array([0,0]),np.array([0,0]),np.array([0,0],bool))
    first=add_features(c.copy(),np.array([[5.,1.],[6.,1.],[7.,1.]]),np.array([1.,2.,3.]),*args)
    changed=add_features(c.copy(),np.array([[5.,1.],[60.,50.],[70.,60.]]),np.array([1.,2.,3.]),*args)
    assert set(FEATURES).issubset(first.columns)
    assert np.allclose(first[FEATURES].to_numpy(),changed[FEATURES].to_numpy())
    assert 'dist_next' not in FEATURES

def test_observation_top1_and_binary_metrics_count_empty_candidate_sets():
    d=pd.DataFrame({'trip_id':['a','a'],'obs_idx':[0,0],'arc_id':[2,3],'ref_arc_id':[2,2]})
    obs=pd.DataFrame({'trip_id':['a','a'],'obs_idx':[0,1]})
    assert top1(d,np.array([.4,.8]),obs)==0.0
    metrics=row_metrics(np.array([0,1]),np.array([.1,.9]))
    assert metrics['macro_f1']==1.0

def test_clean_track_retains_original_observation_indices():
    raw=pd.DataFrame({'obs_idx':[0,1,2],'lon':[1.,2.,3.],'lat':[1.,2.,3.],'t':[2.,1.,3.]})
    cleaned=clean_track(raw)
    assert cleaned.obs_idx.tolist()==[1,0,2]

def test_segment_split_at_large_time_gap():
    ids=split_segments(np.array([[0.,0.],[1.,0.],[2.,0.]]),np.array([0.,1.,100.]))
    assert ids.tolist()==[0,0,1]
