'''
upper,lower,
mylist= [0.0, 0.4, 0.81, 1.0, 0.9, 20.7, 0.0, 0.8, 1.0, 20.7]
sorted_mylist = sorted(((v, i) for i, v in enumerate(mylist)), reverse=True)
str1 = input()
'''


# If you need to import additional packages or classes, please import here.

import numpy as np

def GetValue(arr,ss):
    sum1 = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            sum1 = sum1 + ss[i]
    return sum1

def func():

    str1 = input()
    arr = str.split(str1," ")

    M = int(arr[0])
    prices = int(arr[1])
    count = int(arr[2])

    arr = []
    str2 = input()
    bb = str.split(str2," ")
    for i in range(count):
        a = int(bb[i])
        arr.append(a)

    r = 0
    for i in range(count):
        sum1 = prices
        sum2 = prices
        sum3 = prices

        mycount = 0
        for j in range(count):
            if j != i:
                sum1 = sum1 + arr[j]
                mycount = mycount + 1
            if mycount == M:
                break

        if sum1 == 0:
            r = r + 1
    
    print(r)

    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().

if __name__ == "__main__":
    lst = [0,1,2,2,3,4,4,4,5,6]
    lst = np.array(lst)
    print(np.argmax(np.bincount(lst)))
