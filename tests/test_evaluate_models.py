import numpy as np
import pytest

from scripts.evaluate_models import compute_metrics, _valid_class_curves

def test_binary_metrics_and_auc_use_probabilities():
    y=np.array([0,0,1,1]); p=np.array([[.9,.1],[.3,.7],[.4,.6],[.1,.9]])
    m=compute_metrics(y,np.array([0,1,1,1]),p,np.array([0,1]))
    assert m['accuracy']==.75
    assert m['roc_auc']==pytest.approx(.75)
    assert m['roc_auc_macro']==pytest.approx(.75)
    assert m['roc_auc_micro']==pytest.approx(.8125)
    assert m['log_loss']>0

def test_multiclass_ovr_macro_micro_and_missing_class_are_safe():
    classes=np.array(['a','b','c']); y=np.array(['a','b','a','b'])
    p=np.array([[.8,.1,.1],[.1,.8,.1],[.7,.2,.1],[.2,.7,.1]])
    curves,per_auc,micro,macro,_=_valid_class_curves(y,p,classes)
    assert set(curves)=={'a','b'}
    assert 'c' not in per_auc
    assert np.isfinite(micro[2]) and np.isfinite(macro[2])
    m=compute_metrics(y,np.array(['a','b','a','b']),p,classes)
    assert m['classes_in_test']==2
    assert m['undefined_roc_classes']=='["c"]'

def test_single_class_test_split_marks_roc_undefined():
    m=compute_metrics(np.array([0,0]),np.array([0,0]),np.array([[.9,.1],[.8,.2]]),np.array([0,1]))
    assert np.isnan(m['roc_auc'])
    assert np.isnan(m['roc_auc_macro'])
    assert np.isnan(m['roc_auc_micro'])

@pytest.mark.parametrize('y,pred,p,classes',[ 
    ([0,1],[0],[[.5,.5],[.5,.5]],[0,1]),
    ([0,1],[0,1],[[.5],[.5]],[0,1]),
    ([0,1],[0,1],[[np.nan,.0],[.2,.8]],[0,1]),
    ([0,1],[0,1],[[.2,.2],[.3,.7]],[0,1]),
    ([0,1],[0,1],[[1.2,-.2],[.3,.7]],[0,1]),
    ([0,2],[0,2],[[.8,.2],[.2,.8]],[0,1]),
])
def test_invalid_inputs_raise(y,pred,p,classes):
    with pytest.raises(ValueError): compute_metrics(np.asarray(y),np.asarray(pred),np.asarray(p),np.asarray(classes))
