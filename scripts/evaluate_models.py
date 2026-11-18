"""Recompute MLP test metrics and plots from a saved artifact, without fitting."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_curve)
from sklearn.preprocessing import label_binarize

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"
ARTIFACTS=ROOT/"artifacts"/"models"
PRED_DIR=ROOT/"results"/"preds"
OUT=ROOT/"results"/"metrics"
PLOTS=ROOT/"results"/"plots"
# KNN remains available in historical artifacts and source, but is no longer part
# of the active evaluation workflow.
MODEL_NAMES=("mlp",)
ID_COLUMNS=["trip_id","obs_idx","arc_id","ref_arc_id","label"]

def read_table(stem: str, data_dir=DATA) -> pd.DataFrame:
    p=Path(data_dir)/f"{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(Path(data_dir)/f"{stem}.csv",dtype={"trip_id":str})

def read_artifact_table(stem: str, pred_dir=PRED_DIR) -> pd.DataFrame:
    p=Path(pred_dir)/f"{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(Path(pred_dir)/f"{stem}.csv",dtype={"trip_id":str})


def validate_saved_predictions(saved_pred, test, model_scores):
    """Ensure predictions are row-aligned with the evaluated data and model probabilities."""
    if len(saved_pred) != len(test) or not all(c in saved_pred for c in ID_COLUMNS + ["score"]):
        raise ValueError("saved predictions have an incompatible schema or row count")
    if not saved_pred[ID_COLUMNS].reset_index(drop=True).equals(test[ID_COLUMNS].reset_index(drop=True)):
        raise ValueError("saved prediction rows do not match current test IDs and labels")
    if not np.allclose(saved_pred.score.to_numpy(float), np.asarray(model_scores, float), rtol=1e-5, atol=1e-7):
        raise ValueError("saved prediction probabilities do not match the loaded model")

def _valid_class_curves(y_true, scores, classes):
    """Return valid one-vs-rest curves and micro/macro AUC; skip undefined classes safely."""
    y_bin=label_binarize(y_true,classes=classes)
    if len(classes)==2 and y_bin.shape[1]==1:
        y_bin=np.column_stack([1-y_bin[:,0],y_bin[:,0]])
    curves={}; per_auc={}
    if len(np.unique(y_true))<2:
        return curves,per_auc,(None,None,float('nan')),(None,None,float('nan')),y_bin
    for i,label in enumerate(classes):
        target=y_bin[:,i]
        if target.min()==target.max(): continue
        fpr,tpr,_=roc_curve(target,scores[:,i]); curves[label]=(fpr,tpr); per_auc[label]=float(auc(fpr,tpr))
    flat=y_bin.ravel()
    if flat.min()!=flat.max():
        fpr_micro,tpr_micro,_=roc_curve(flat,scores.ravel()); micro=float(auc(fpr_micro,tpr_micro))
    else:
        fpr_micro=tpr_micro=None; micro=float("nan")
    if curves:
        grid=np.unique(np.concatenate([v[0] for v in curves.values()]))
        mean_tpr=np.mean([np.interp(grid,fpr,tpr) for fpr,tpr in curves.values()],axis=0)
        # Macro ROC-AUC is the arithmetic mean of defined per-class OvR AUCs.
        macro=float(np.mean(list(per_auc.values())))
    else:
        grid=mean_tpr=None; macro=float("nan")
    return curves,per_auc,(fpr_micro,tpr_micro,micro),(grid,mean_tpr,macro),y_bin

def compute_metrics(y_true, y_pred, probabilities, classes):
    """Compute classification and OvR ROC metrics from class-aligned probabilities."""
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); probabilities=np.asarray(probabilities,float); classes=np.asarray(classes)
    if y_true.ndim!=1 or y_pred.shape!=y_true.shape: raise ValueError("y_true and y_pred must be 1-D arrays of equal length")
    if probabilities.ndim!=2 or probabilities.shape!=(len(y_true),len(classes)): raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if len(y_true)==0 or not np.isfinite(probabilities).all(): raise ValueError("evaluation data must be non-empty and probabilities finite")
    if np.any(probabilities<0) or np.any(probabilities>1): raise ValueError("probabilities must be in [0, 1]")
    row_sums=probabilities.sum(axis=1)
    if not np.allclose(row_sums,1.,atol=1e-5): raise ValueError("each probability row must sum to 1")
    # Normalize harmless floating-point row-sum drift before Log Loss.
    probabilities=probabilities/row_sums[:,None]
    if len(np.unique(classes))!=len(classes): raise ValueError("classes must be unique")
    if not set(np.unique(y_true)).issubset(set(classes)): raise ValueError("y_true contains labels absent from model classes")
    if not set(np.unique(y_pred)).issubset(set(classes)): raise ValueError("y_pred contains labels absent from model classes")
    curves,per_auc,micro,macro,ybin=_valid_class_curves(y_true,probabilities,classes)
    present=np.unique(y_true)
    # Log loss accepts the full model class ordering, including classes absent in this split.
    try: ll=float(log_loss(y_true,probabilities,labels=classes))
    except ValueError: ll=float("nan")
    auc_binary=per_auc.get(classes[-1],float("nan")) if len(classes)==2 else float("nan")
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_precision":float(precision_score(y_true,y_pred,labels=classes,average="macro",zero_division=0)),
        "macro_recall":float(recall_score(y_true,y_pred,labels=classes,average="macro",zero_division=0)),
        "macro_f1":float(f1_score(y_true,y_pred,labels=classes,average="macro",zero_division=0)),
        "weighted_precision":float(precision_score(y_true,y_pred,labels=classes,average="weighted",zero_division=0)),
        "weighted_recall":float(recall_score(y_true,y_pred,labels=classes,average="weighted",zero_division=0)),
        "weighted_f1":float(f1_score(y_true,y_pred,labels=classes,average="weighted",zero_division=0)),
        "micro_f1":float(f1_score(y_true,y_pred,labels=classes,average="micro",zero_division=0)),
        "roc_auc":auc_binary,"roc_auc_macro":macro[2],"roc_auc_micro":micro[2],"log_loss":ll,
        "classes_in_test":len(present),"classes_in_model":len(classes),
        "undefined_roc_classes":json.dumps([str(x) for x in classes if x not in per_auc]),
    }

def plot_roc(model_name,y_true,probabilities,classes,path):
    curves,per_auc,micro,macro,_=_valid_class_curves(np.asarray(y_true),np.asarray(probabilities),np.asarray(classes))
    fig,ax=plt.subplots(figsize=(8,6))
    for label,(fpr,tpr) in curves.items():
        ax.plot(fpr,tpr,lw=1.6,label=f"Class {label} (AUC={per_auc[label]:.3f})")
    if micro[0] is not None: ax.plot(micro[0],micro[1],lw=2.2,ls=':',label=f"Micro-average (AUC={micro[2]:.3f})")
    if macro[0] is not None: ax.plot(macro[0],macro[1],lw=2.2,ls='--',label=f"Macro-average (AUC={macro[2]:.3f})")
    ax.plot([0,1],[0,1],color='grey',lw=1.2,ls='--',label='Random classifier (AUC=0.500)')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False Positive Rate',ylabel='True Positive Rate',title=f'{model_name.upper()} One-vs-Rest ROC')
    ax.grid(alpha=.25); ax.legend(loc='lower right',fontsize=8); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def plot_confusion(name,y_true,y_pred,classes,path):
    cm=confusion_matrix(y_true,y_pred,labels=classes)
    fig,ax=plt.subplots(figsize=(5.5,4.5)); im=ax.imshow(cm,cmap='Blues')
    ax.set(xticks=np.arange(len(classes)),yticks=np.arange(len(classes)),xticklabels=[str(c) for c in classes],yticklabels=[str(c) for c in classes],xlabel='Predicted class',ylabel='True class',title=f'{name.upper()} confusion matrix')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,f'{v:,}',ha='center',va='center',color='black')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)
    return cm

def evaluate_model(name,test,manifest,train_rows,artifact_dir=ARTIFACTS,pred_dir=PRED_DIR,out_dir=OUT,plots_dir=PLOTS,feature_list_path=None):
    artifact_path=Path(artifact_dir)/f"{name}.joblib"
    if not artifact_path.exists(): raise FileNotFoundError(f"Missing saved model {artifact_path}; train {name} first")
    saved=joblib.load(artifact_path); model=saved["model"]; features=list(saved["features"])
    feature_path=Path(feature_list_path) if feature_list_path else DATA/"feature_list.json"
    if not feature_path.exists(): feature_path=ROOT/"artifacts"/"feature_list.json"
    if features!=json.loads(feature_path.read_text()): raise ValueError(f"{name} artifact feature schema differs from current feature_list.json")
    absent=[c for c in features if c not in test]
    if absent: raise ValueError(f"Student-test table lacks {name} feature columns: {absent}")
    X=test[features].to_numpy(np.float32); y=test.label.to_numpy(); estimator=model[-1] if hasattr(model,"steps") else model; classes=np.asarray(estimator.classes_)
    if not hasattr(model,"predict_proba"): raise TypeError(f"{name} artifact has no predict_proba; ROC requires probability scores")
    start=time.perf_counter(); raw=model.predict_proba(X); infer_s=time.perf_counter()-start
    if raw.shape[1]!=len(classes): raise ValueError(f"{name} predict_proba columns do not match estimator classes")
    if not set(np.unique(y)).issubset(set(classes)): raise ValueError(f"{name} model class set does not cover student-test labels")
    positive_col=list(classes).index(1) if 1 in classes else int(np.argmax(classes))
    saved_pred=read_artifact_table(f"{name}_student_test", pred_dir)
    validate_saved_predictions(saved_pred, test, raw[:,positive_col])
    pred=classes[np.argmax(raw,axis=1)]
    result=compute_metrics(y,pred,raw,classes)
    # Road-selection Top-1 is distinct from candidate-row classification accuracy.
    score=raw[:,list(classes).index(1)] if 1 in classes else raw.max(axis=1)
    scored=test[["trip_id","obs_idx","arc_id","ref_arc_id"]].copy(); scored['score']=score
    best=scored.loc[scored.groupby(['trip_id','obs_idx'],sort=False).score.idxmax()]
    correct={(r.trip_id,int(r.obs_idx)):int(r.arc_id==r.ref_arc_id) for r in best.itertuples()}
    result['top1_accuracy']=float(np.mean([correct.get((r.trip_id,int(r.obs_idx)),0) for r in manifest.itertuples()]))
    result['n_train_samples']=int(train_rows); result['n_test_candidate_rows']=int(len(test)); result['n_test_observations']=int(len(manifest))
    result['inference_time_s']=float(infer_s); result['inference_time_s_per_1000_rows']=float(infer_s*1000/max(len(test),1))
    report=classification_report(y,pred,labels=classes,target_names=[str(c) for c in classes],output_dict=True,zero_division=0)
    report_df=pd.DataFrame(report).T; report_df.index.name='class_or_average'; report_df.to_csv(Path(out_dir)/f"classification_report_{name}.csv")
    cm=plot_confusion(name,y,pred,classes,Path(plots_dir)/f"confusion_matrix_{name}.png")
    pd.DataFrame(cm,index=[f'true_{c}' for c in classes],columns=[f'pred_{c}' for c in classes]).to_csv(Path(out_dir)/f"confusion_matrix_{name}.csv")
    plot_roc(name,y,raw,classes,Path(plots_dir)/f"roc_curve_{name}.png")
    result['model']=name
    return result

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="", help="versioned run name, e.g. post_correction")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--prediction-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()
    tag = args.run_tag
    data_dir = Path(args.data_dir) if args.data_dir else (DATA/tag if tag else DATA)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else (ARTIFACTS/tag if tag else ARTIFACTS)
    pred_dir = Path(args.prediction_dir) if args.prediction_dir else (ROOT/"results"/tag/"preds" if tag else PRED_DIR)
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT/"results"/tag/"metrics" if tag else OUT)
    plots_dir = Path(args.plots_dir) if args.plots_dir else (ROOT/"results"/tag/"plots" if tag else PLOTS)
    for directory in (data_dir, artifact_dir, pred_dir, out_dir, plots_dir):
        if not directory.is_absolute():
            directory = ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
    # Resolve relative paths consistently after creating their directories.
    data_dir = data_dir if data_dir.is_absolute() else ROOT/data_dir
    artifact_dir = artifact_dir if artifact_dir.is_absolute() else ROOT/artifact_dir
    pred_dir = pred_dir if pred_dir.is_absolute() else ROOT/pred_dir
    out_dir = out_dir if out_dir.is_absolute() else ROOT/out_dir
    plots_dir = plots_dir if plots_dir.is_absolute() else ROOT/plots_dir
    test=read_table('candidates_student_test', data_dir); manifest=read_table('observations_student_test', data_dir)
    sample_cfg=data_dir/'training_sample_config.json'
    train_rows=json.loads(sample_cfg.read_text())['actual_rows'] if sample_cfg.exists() else len(read_table('candidates_train', data_dir))
    rows=[evaluate_model(n,test,manifest,train_rows,artifact_dir,pred_dir,out_dir,plots_dir,
                         feature_list_path=data_dir/"feature_list.json") for n in MODEL_NAMES]
    metrics=pd.DataFrame(rows).set_index('model')
    metrics.to_csv(out_dir/'model_metrics.csv')
    candidate_sets={(tid,int(obs)):set(g.arc_id.astype(int)) for (tid,obs),g in test.groupby(['trip_id','obs_idx'])}
    misses=[int(r.ref_arc_id) not in candidate_sets.get((r.trip_id,int(r.obs_idx)),set()) for r in manifest.itertuples()]
    miss=float(np.mean(misses)) if misses else float('nan')
    candidate_stats=pd.read_csv(data_dir/'candidate_stats_student_test.csv')
    pd.DataFrame([{'n_student_test_observations':len(manifest),'candidate_miss_rate':miss,'candidate_coverage':1-miss,
                   'observations_without_candidates':sum((r.trip_id,int(r.obs_idx)) not in candidate_sets for r in manifest.itertuples()),
                   'mean_candidates_per_observation':len(test)/max(len(manifest),1),
                   'mean_trip_candidate_miss_rate_pct':float(candidate_stats.candidate_miss_rate.mean()) if 'candidate_miss_rate' in candidate_stats else np.nan}]).to_csv(out_dir/'candidate_generation_metrics.csv',index=False)
    # Keep multiclass/undefined-class behavior and metric definitions alongside results.
    (out_dir/'evaluation_notes.json').write_text(json.dumps({'selected_classifier':'MLP','discontinued_classifier':'KNN; historical code and results retained, excluded from active evaluation','classification_target':'binary candidate correctness','roc':'one-vs-rest from saved predict_proba values; macro averages defined class curves; micro pools one-hot decisions','missing_classes':'classes with no positive or no negative examples have no individual ROC curve and are listed in undefined_roc_classes; macro averages valid class curves only','top1':'mean over the full supplied observation manifest; highest probability candidate correctness per observation; observations with no candidates count wrong','candidate_generation_metrics':'saved separately in candidate_generation_metrics.csv'},indent=2)+'\n')
    print(metrics[['accuracy','top1_accuracy','macro_f1','roc_auc','roc_auc_macro','roc_auc_micro','log_loss']].to_string())
    print(f"Saved MLP metrics and reports in {out_dir}; plots in {plots_dir}")

if __name__=='__main__': main()
