import numpy as np

def numpy_alternate_sort(array):
    sorted_array = np.sort(array)
    result = []
    left = 0
    right = len(sorted_array) - 1
    
    while left <= right:
        result.append(sorted_array[left])
        result.append(sorted_array[right])
        left += 1
        right -= 1
    
    return np.array(result)

a=np.random.uniform(0,10,20)

print(a)
print(np.around(a,2))

print("Minimum is",np.min(a))
print("Maximum is",np.max(a))
print("Mean is",np.mean(a))

a=[x*x for x in a if x<5]
print(a)
print(np.random.randint(50,100))