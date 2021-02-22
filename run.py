import numpy as np
from sklearn import svm
from sklearn import model_selection as ms
from tqdm import tqdm
from warnings import filterwarnings
import binary_optimization as opt
filterwarnings('ignore')


with open("wine/0.6/wine_train_data_testrate0.6.txt") as f:
    tr_d=np.array([[float(d) for d  in data.split(',')] for data in f.read().splitlines()])
    
with open("wine/0.6/wine_test_data_testrate0.6.txt") as f:
    te_d=np.array([[float(d) for d  in data.split(',')] for data in f.read().splitlines()])

with open("wine/0.6/wine_train_label_testrate0.6.txt") as f:
    tr_l=np.array([int(data) for data in f.read().splitlines()])

with open("wine/0.6/wine_test_label_testrate0.6.txt") as f:
    te_l=np.array([int(data) for data in f.read().splitlines()])

def test_score(gen,tr_x,tr_y,te_x,te_y):
    clf = svm.LinearSVC()
    mask=np.array(gen) == 1
    al_data=np.array(tr_x[:,mask])
    al_test_data=np.array(te_x[:,mask])
    return np.mean([svm.LinearSVC().fit(al_data,tr_y).score(al_test_data,te_y) for i in range(4)])

class Evaluate:
    def __init__(self):
        self.train_l = tr_l
        self.train_d = tr_d
        self.K = 4
    def evaluate(self,gen):
        mask=np.array(gen) > 0
        al_data=np.array([al[mask] for al in self.train_d])
        kf = ms.KFold(n_splits=self.K)
        s = 0
        for tr_ix,te_ix in kf.split(al_data):
            s+= svm.LinearSVC().fit(al_data[tr_ix],self.train_l[tr_ix]).score(al_data[te_ix],self.train_l[te_ix])#.predict(al_test_data)
        s/=self.K
        return s
    def check_dimentions(self,dim):
        if dim==None:
            return len(self.train_d[0])
        else:
            return dim


results = {}

max_iters = 20
number_of_particles = 20
results["BGA"] = opt.BGA(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BPSO"] = opt.BPSO(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BCS"] = opt.BCS(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BFFA"] = opt.BFFA(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BBA"] = opt.BBA(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BGSA"] = opt.BGSA(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)
results["BDFA"] = opt.BDFA(Eval_Func=Evaluate,n=number_of_particles,m_i=max_iters)


print("Algorithm:\n\t{0}         {1}      {2}      {3}".format("best_features","best_val","number_of_1s", "test_score"))

for key in results.keys():
    s, g, l, a = results[key]
    print("{0}:\n\t{1}          {2:.6f}           {3}             {4:.6f}".format(str(key), "".join(map(str,g)),s,l, test_score(g,tr_d,tr_l,te_d,te_l)))

# HTML(results["BGA"][3].to_html5_video())

# HTML(results["BPSO"][3].to_html5_video())

# HTML(results["BCS"][3].to_html5_video())

# HTML(results["BFFA"][3].to_html5_video())

# HTML(results["BBA"][3].to_html5_video())

# HTML(results["BGSA"][3].to_html5_video())

# HTML(results["BDFA"][3].to_html5_video())