from sklearn import preprocessing
import numpy as np

a = np.random.random((1, 4))
a=[0,5,10]*a +b =[b,5a+b,10a+b]
N~(0,1)*a+b -> N~(b,a)
E(a)= b
std=a
a.mean()
a.std()
print("Data before = ", a,a.mean(),a.std())
# normalized = preprocessing.normalize(a)
a_norm = (a - a.mean()) / a.std()
print("Normalized Data = ", a_norm,a_norm.mean(),a_norm.std())

a.mean()
a.std()
print("Data now = ", a,a.mean(),a.std())

# normalize the data attributes
a_norm = (a - a.mean()) / a.std()
a_norm = a_norm*-2
print("Normalized Data = ", a_norm,a_norm.mean(),a_norm.std())