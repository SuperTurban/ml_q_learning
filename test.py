from collections import deque
import random

a = deque()

a.append(2)
a.append(3)
a.append(5)
a.append(12)
a.append(11)
a.append(122)
a.append(1)
a.append(142)
a.append(12)
a.append(11)
a.append(122)
a.append(1)

b = random.sample(range(len(a)), 10)
print(b)
for i in b:
    print(a[i])

print(random.rand())


