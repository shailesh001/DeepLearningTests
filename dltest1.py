import numpy as np

a = np.random.randn(5)
print(a)

expa = np.exp(a)
print(expa)

#
# Create the softmax probabilities
print(expa.sum())
answer = expa / expa.sum()
print(answer)

# Check they add up to 1
print(answer.sum())

# Now create a 2-D array
A = np.random.randn(100, 5)
print(A)

expA = np.exp(A)
print(expA)

# Divide each item by the sum of each row
# Only sum each row not the entire array.
# keepdims allows a 1-D array to divide into a 2-D array by keeping the array shape to 2-D for array
answerA = expA / expA.sum(axis=1, keepdims=True)
print(answerA)

# Check the sum of each row
print(answerA.sum(axis=1))

# Check the shape
print(expA.sum(axis=1, keepdims=True))

# Dhow that the shape of the sum'd array is 2-D even though the list item within the list only contains 1 item
print(expA.sum(axis=1, keepdims=True).shape)


