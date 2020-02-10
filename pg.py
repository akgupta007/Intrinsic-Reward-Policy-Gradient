import numpy as np
import gym

H = 20 
H1 = 10
batch_size = 10 
learning_rate = 1e-4
lamda = 0.5
gamma = 0.99 
decay_rate = 0.99 
render = True

D = 128 
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D)
model['W2'] = np.random.randn(H) / np.sqrt(H)
reward_model = {}
reward_model['W1'] = np.random.randn(H1,D+1) / np.sqrt(D+1)
reward_model['W2'] = np.random.randn(H1) / np.sqrt(H1)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } 

reward_grad_buffer = { k : np.zeros_like(v) for k,v in reward_model.items() }
reward_rmsprop_cache = { k : np.zeros_like(v) for k,v in reward_model.items() }


def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) 

def discounted_i_reward(r1, r):
  discounted_r = np.zeros_like(r)
  running_add = np.zeros(r.shape[1])
  for t in reversed(range(0, r1.size)):
    if r1[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t,:]
    discounted_r[t,:] = running_add
  return discounted_r

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  # print (r.shape)
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def state_reward_forward(x):
  h = np.dot(reward_model['W1'], x)
  h[h<0] = 0 
  logp = np.dot(reward_model['W2'], h)
  p = sigmoid(logp)
  return p, h 

def state_reward_backward(epx, eph, epdlogp):
  eph = np.reshape(eph, (1, eph.shape[0]))
  epx = np.reshape(epx, (1, epx.shape[0]))
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, reward_model['W2'])
  dh[eph <= 0] = 0
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h

def policy_backward_w1(epx, eph, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).T
  dh = np.outer(epdlogp.T, model['W2'])
  dh = np.reshape(dh, (epdlogp.shape[1],epdlogp.shape[0],model['W2'].shape[0]))
  dh[eph <= 0] = 0
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def policy_backward(epx, eph, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-ram-v0")
observation = env.reset()
prev_x = None
xs,hs,dlogps,drs = [],[],[],[]
drs_i = []
gra_reward_w1 = []
gra_reward_w2 = []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  cur_x = observation
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3

  xs.append(x)
  hs.append(h)
  y = 1 if action == 2 else 0
  dlogps.append(y - aprob)

  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward)
  saPair = np.append(x, action)
  i_reward, intrinsic_h = state_reward_forward(saPair)
  drs_i.append(reward + lamda*i_reward)
  
  back_prop_reward = i_reward * (1- i_reward)
  gra_reward = state_reward_backward(saPair, intrinsic_h, back_prop_reward)
  gra_reward_w1.append(gra_reward['W1'].ravel())
  gra_reward_w2.append(gra_reward['W2'].ravel())


  if done:
    episode_number += 1

    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epdlogp_i = epdlogp
    epr = np.vstack(drs)
    epr_i = np.vstack(drs_i)

    egra_reward_w1 = np.vstack(gra_reward_w1)
    egra_reward_w2 = np.vstack(gra_reward_w2)
    xs,hs,dlogps,drs = [],[],[],[]
    drs_i = []
    gra_reward_w1, gra_reward_w2 = [],[]

    discounted_epr = discount_rewards(epr)
    discounted_epr_i = discount_rewards(epr_i)
    discounted_reward_gradient_w1 = discounted_i_reward(epr, egra_reward_w1)
    discounted_reward_gradient_w2 = discounted_i_reward(epr, egra_reward_w2)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    discounted_epr_i -= np.mean(discounted_epr_i)
    discounted_epr_i /= np.std(discounted_epr_i)

    grad_n = {}
    grad_n['W1'] = np.zeros((H1, D+1))
    grad_n['W2'] = np.zeros((H1,))
    epdlogp_w1 = epdlogp * discounted_reward_gradient_w1
    epdlogp_w2 = epdlogp * discounted_reward_gradient_w2
    epdlogp *= discounted_epr
    epdlogp_i *= discounted_epr_i
    grad1 = policy_backward(epx, eph, epdlogp)
    grad = policy_backward(epx, eph, epdlogp_i)
    for i in range(epdlogp_w2.shape[1]):
      grad_w2 = policy_backward(epx, eph, np.vstack(epdlogp_w2[:,i]))
      grad_n['W2'][i] = np.sum(grad_w2['W1'] * grad1['W1'])
      grad_n['W2'][i] += np.sum(grad_w2['W2'] * grad1['W2'])
    
    # for i in range(H1):
      # print (i)
    for j in range(H1*(D+1)):
      # print (j)
      grad_w1 = policy_backward(epx, eph, np.vstack(epdlogp_w1[:,j]))  # change
      pos_x = int(j/(D+1))
      pos_y = j%(D+1)
      grad_n['W1'][pos_x][pos_y] = np.sum(grad_w1['W1'] * grad1['W1'])
      grad_n['W1'][pos_x][pos_y] += np.sum(grad_w1['W2'] * grad1['W2'])

    for k in model: grad_buffer[k] += grad[k]
    for k in model: reward_grad_buffer[k] += grad_n[k]

    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k]
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v)

      for k,v in reward_model.items():
        g = reward_grad_buffer[k]
        reward_rmsprop_cache[k] = decay_rate * reward_rmsprop_cache[k] + (1 - decay_rate) * g**2
        reward_model[k] += learning_rate * g / (np.sqrt(reward_rmsprop_cache[k]) + 1e-5)
        reward_grad_buffer[k] = np.zeros_like(v)

    reward_sum = 0
    observation = env.reset()
    prev_x = None
