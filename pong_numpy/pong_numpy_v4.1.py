import numpy as np
import pickle
import gym

render = True
resume = True

def prepro(I):
  I = I[35:195]
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(float).ravel()

def initialisation(n0, n1, n2):
  W1 = np.random.randn(n1, n0) / np.sqrt(n0) # "Xavier" initialization
  B1 = np.random.randn(n1, 1) / np.sqrt(n1)
  W2 = np.random.randn(n2, n1) / np.sqrt(n1)
  B2 = np.random.randn(n2, 1) / np.sqrt(n2)

  parameters = {
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2
  }

  return parameters

def ReLU(X):
  X[X<0] = 0
  return X

def Sigmoid(X):
  return 1/(1+np.exp(-X))

def dérivée_ReLU(X):
  X[X<=0] = 0
  X[X>0] = 1
  return X

def forward_prop(X, parameters):
  W1, B1, W2, B2 = parameters['W1'], parameters['B1'], parameters['W2'], parameters['B2']

  X = X.reshape(X.shape[0], 1)

  Z1 = W1@X+B1
  A1 = ReLU(Z1)

  Z2 = W2@A1+B2
  A2 = Sigmoid(Z2)

  A1 = A1.reshape(200,)
  A2 = A2.reshape(1,)

  return A1, A2

def backprop(X, A1, A2, Y, discounted_epr, parameters):
  W2 = parameters['W2']

  m = Y.shape[0]

  dZ2 = (A2 - Y)*discounted_epr.T
  dW2 = dZ2 @ A1.T
  dB2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

  dZ1 = W2.T @ dZ2 * dérivée_ReLU(A1)
  dW1 = dZ1 @ X.T
  dB1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

  gradients = {
    'dW1': dW1,
    'dB1': dB1,
    'dW2': dW2,
    'dB2': dB2
  }

  return gradients

def update(parameters, gradients, lr):
  W1, B1, W2, B2 = parameters['W1'], parameters['B1'], parameters['W2'], parameters['B2']
  dW1, dB1, dW2, dB2 = gradients['dW1'], gradients['dB1'], gradients['dW2'], gradients['dB2']

  W1 -= lr*dW1
  B1 -= lr*dB1
  W2 -= lr*dW2
  B2 -= lr*dB2

  parameters = {
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2
  }

  return parameters

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * 0.99 + r[t]
    discounted_r[t] = running_add
  return discounted_r

if resume:
  parameters = pickle.load(open('pong_numpy/sauvegarde_900.p', 'rb'))
else:
  parameters = initialisation(6400, 200, 1)


env = gym.make("Pong-v0")
observation = env.reset()
lX, lA1, lA2, ly, lr = [], [], [], [], []
running_reward = -21   # correspond à une défaite totale
reward_sum = 0
episode_number = 0


while True:
  if render: env.render()

  X = prepro(observation)

  A1, A2 = forward_prop(X, parameters)
  y = 1 if np.random.uniform() < A2 else 0

  observation, reward, done, _ = env.step(y + 2)    # y+2 donne l'action à effectuer (monter ou descendre) (voir documentation gym)
  reward_sum += reward

  lX.append(X)
  lA1.append(A1)
  lA2.append(A2)
  ly.append(y)
  lr.append(reward)


  if done:
    episode_number += 1

    epX = np.stack(lX, axis=1)
    epA1 = np.stack(lA1, axis=1)
    epA2 = np.stack(lA2, axis=1)
    epy = np.array(ly); epy = epy.reshape(1, epy.shape[0])
    epr = np.vstack(lr)
    lX, lA1, lA2, ly, lr = [], [], [], [], []

    discounted_epr = discount_rewards(epr)
    discounted_epr = (discounted_epr - np.mean(discounted_epr)) / np.std(discounted_epr)

    gradients = backprop(epX, epA1, epA2, epy, discounted_epr, parameters)
    parameters = update(parameters, gradients, lr=1e-3)

    running_reward = running_reward * 0.99 + reward_sum * 0.01
    print(f"episode: {episode_number}     episode reward: {reward_sum}     running reward: {running_reward}")

    if episode_number % 50 == 0:
      print(f"episode {episode_number}, saving model ...")
      pickle.dump(parameters, open('sauvegarde.p', 'wb'))

    reward_sum = 0
    observation = env.reset()