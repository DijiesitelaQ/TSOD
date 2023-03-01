import torch


a = torch.ones([2, 3])
print(a)
c = torch.randn([2, 3])
print(c)
b = torch.add(a, c,alpha=1.3)
print(b)
# c = torch.randn([2, 3])
# print( c)
# d = torch.add(a, c)
# print (d)


