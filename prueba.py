import scipy.io
from sklearn.model_selection import train_test_split

x=[]
path="/mnt/c/users/santi/Data_simulation/x/"

for elementos in range(0,1999):
    data = scipy.io.loadmat("{}X_{}.mat".format(path, elementos))
    epsil = data['epsil']
    x.append(epsil)

y=x