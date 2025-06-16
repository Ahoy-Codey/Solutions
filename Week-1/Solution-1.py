import numpy as np


def  numpy_boundary_traversal(matrix):
    top=matrix[0,:]
    right=matrix[1:,-1]
    bottom=matrix[-1,:-1][::-1]
    left=matrix[1:-1,0][::-1]

    return np.concat([top,right,bottom,left]).tolist()
a=np.random.randint(1,51,size=(5,4))
print(a)


ad=[a[i,3-i] for i in range(min(5,4))]

print(ad)

mar=np.max(a,axis=1)

print(mar)

am=np.mean(ad)

b=a[a>am]

print(f"Arithmetic mean is {am}")

print("resultant array is",b)

print(f"Boundaries elements are {numpy_boundary_traversal(a)}")