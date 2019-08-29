import numpy as np
import time

pred_1=np.random.normal(size=[5, 10, 5, 1, 10])
samples_1=np.random.normal(size=[5, 10, 5, 6, 10])
samples_1=np.concatenate((samples_1, pred_1), axis=-2)

def dotProductManual(vec1, vec2):
    norm_vec1=np.linalg.norm(vec1, axis=-1)
    norm_vec2=np.linalg.norm(vec2, axis=-1)
    print(norm_vec1.shape)
    print(norm_vec2.shape)

def dotProduct(vec1, vec2):
    result=np.mean(vec1*vec2, axis=-1)

    print(result.shape)
    print(np.exp(result[0,0,0])/np.sum(np.exp(result[0,0,0])))

start_t=time.time()
dotProductManual(pred_1, samples_1)
print('Manual in {:.2f}s'.format(time.time()-start_t))

start_t=time.time()
dotProduct(pred_1, samples_1)
print('Auto in {:.2f}s'.format(time.time()-start_t))