"""Factories for comparable binary candidate classifiers."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def make_mlp(seed=42, max_iter=60):
    """Unfitted MLP factory; iteration count is selected with trip-grouped validation."""
    return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=1e-4,batch_size='auto',learning_rate_init=1e-3,max_iter=max_iter,early_stopping=False,random_state=seed))

def make_knn():
    return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=25,weights='distance',metric='minkowski',p=2,algorithm='auto',n_jobs=-1))
