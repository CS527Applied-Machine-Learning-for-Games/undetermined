from __future__ import division

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio
import scipy.misc
import os
from PIL import ImageFile
import time
import traceback

# Actions x y in domain of
ACTION_RANGE = range(0,91)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# The scores needed for a reward function
levels1 = [35000, 62000, 43000, 30000, 70000]
levels2 = [64530,64000,105400,60000,90000,80000,70000,70000,60000,55000,90000,80000,85000,59000,60000,70000,60000,70000,65000,60000,95000]
# Your samples location
samples_location = "./Samples/AllAngle/"

class StateMaker():
    def __init__(self):
        # Crops 480x840x3 picture to 310x770x3 and
        # then resizes it to 84x84x3
        # also normalizes the pixel values to -1,1 range
        # Important: pass png without alpha channel
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[480, 840, 3], dtype=tf.float32)
            self.output = tf.image.per_image_standardization(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 80, 20, 310, 770)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def make(self, sess, state):
        return sess.run(self.output, {self.input_state: state})

'''
Class loading expirenced states action pairs

'''
class ExperienceReplay():
    def __init__(self, memory_size=500000):
        self.memory = []
        self.memory_size = memory_size
        self.traj = 4

    # Adds observations to memory
    def remember(self, observation):
        self.memory.extend(observation)

    # Randomly samples memory
    def sample(self, sample_size):
        return np.reshape(random.sample(self.memory, sample_size), [sample_size, 5]) # s,a,r,s',d

    def savesample(self, path = './ddqn_replay_buffer'):
        np.save(path, self.memory)

    def loadsample(self, path = './ddqn_replay_buffer.npy'):
        self.memory = list(np.load(path, allow_pickle=True))

    # Loads memory (screenshots) to its experience
    def load(self):
        #Todo: need change level
        level = 0
        print("Loading experience...")
        for l in range(self.traj):
            path = './Saves/traj' + str(l+1)

            # First read all actions.
            with open(path +  '/actions') as fp:
                actions = fp.readlines()
            actions = [x.strip() for x in actions]
            rewards = []
            # prepare s,a,r,s',d pairs
            count = 0
            s = 'None'
            filename = sorted(os.listdir(path))
            filename.remove('actions')
            try:
                filename.remove('.DS_Store')
            except Exception as e:
                pass
            d = 0
            while True:

                if s == 'None':
                    firstfilename = filename[count]
                    print(firstfilename)
                    s = imageio.imread(path + '/' + firstfilename)[:, :, :3]
                    s = state_maker.make(sess, s)

                    '''
                    if (filename.split('_')[1].split('+')[1] == "None"):
                        a = 0
                    else:
                        a = filename.split('_')[1].split('+')[1]
                    '''

                    r = int(firstfilename.split('_')[1].split('.')[0])

                    # Uncomment next lines if you would like to use Reward Clipping technique
    #                if (r > 3000):
    #                    r = 1
    #                else:
    #                    r = -1
                    r = round( r / levels2[level] ,2) # reward is the ratio from total desired score
                    print(r)
                    count += 1


                nextfilename = filename[count]
                print(nextfilename)
                r = int(nextfilename.split('_')[1].split('.')[0])
                # Uncomment next lines if you would like to use Reward Clipping technique
    #            if (r > 3000):
    #                r = 1
    #            else:
    #                r = -1
                r = round( r / levels2[level] ,2) # reward is the ratio from total desired score

                s1 = imageio.imread(path + '/' + nextfilename)[:, :, :3]
                s1 = state_maker.make(sess, s1)
                print(s1.shape)

                a = actions[count-1]


                if len(nextfilename.split('_')) == 3:
                    d = 1
                    self.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    break
                else:
                    self.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    count += 1
                    s = s1

                '''
                if len(cfilename.split('_')) == 3:
                    d = 1
                    level += 1
                    if cfilename.split('_')[2] == "Won":
                        r *= 1
                    # else:
                    #     r *= -1 #if we lose, reward is extemely negative
                    self.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    s = 'None'
                    if level > 20:
                        level = 0
                else:
                    self.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    s = s1
                '''
            print("Loaded all experience!")





class SummaryStorage():
    def __init__(self, scope="summary", dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            if dir:
                summary_dir = os.path.join(dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

# go through online vars, update the offline vars with online var * update rate + (1-update rate) * offline var
def updateTargetTF(variables, update_rate):
    parameters = []
    for i, variable in enumerate(variables[0:len(variables) // 2]):
        parameters.append(variables[i+len(variables)//2].assign( (variable.value()*update_rate) + ((1-update_rate) * variables[i+len(variables)//2].value()) ))
    return parameters


def updateTarget(parameters, sess):
    for p in parameters:
        sess.run(p)

class DDQN():
    # Model of our agent, follows the original DQN + dueling + double
    # defined in the Nature paper and other Google DeepMind papers
    # Note: out_size here is the size of the last conv layer output
    #
    # More on DQN look at Nature DeepMind paper
    def __init__(self, out_size):
        self.imageIn = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32, name="X")
        self.imageIn = tf.reshape(self.imageIn, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d( inputs=self.imageIn,
                                  num_outputs=32,
                                  kernel_size=[8, 8],
                                  stride=[4, 4],
                                  padding='VALID',
                                  biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=64,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3,
                                 num_outputs=out_size,
                                 kernel_size=[7, 7],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)

        # Dueling DQN implementation
        # Split the output of the last convolution layer to advantage and value
        #
        # More on dueling DQN here: "Dueling Network Architectures for Deep RL" https://arxiv.org/pdf/1511.06581.pdf
        self.advantages_raw, self.values_raw = tf.split(self.conv4, 2, 3) #split 512 to 2 streams
        self.advantages_flatten      = slim.flatten(self.advantages_raw) # 256x1
        self.values_flatten          = slim.flatten(self.values_raw) # 256x1

        # Initialize the weights with xavier
        # More info on why xavier here: proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        xavier = tf.contrib.layers.xavier_initializer()
        self.advantages_weights = tf.Variable(xavier([256, len(ACTION_RANGE)]) ) # 256x90 usefulness of actions
        self.values_weights     = tf.Variable(xavier([256, 1])) # 256x1 usefulness of state
        self.advantages         = tf.matmul(self.advantages_flatten, self.advantages_weights)
        self.values             = tf.matmul(self.values_flatten, self.values_weights)

        # Formula taken from original paper:
        # Q(s,a,delta,alpha,beta) = V(s,delta,beta) + ( A(s,a,delta,alpha)- (1/|A|)*sum_i(A(s,a_i,delta,alpha)) )
        self.q_values = self.values + tf.subtract(self.advantages, tf.reduce_mean(self.advantages, axis=1, keep_dims=True))
        self.best_q = tf.argmax(self.q_values, 1) # get the best one

        # Calculate the loss between target(offline) and online NN, taken from original DQN
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, len(ACTION_RANGE), dtype=tf.float32) # get 1's in choosen action
        self.predicted_q = tf.reduce_sum(tf.multiply(self.q_values, self.actions_onehot), axis=1)

        # RMSOptimization
        self.loss = tf.reduce_mean(tf.square(self.target_q - self.predicted_q))
        self.adam = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.optimized = self.adam.minimize(self.loss)

        # Store summaries
        #Noise addition ?
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("q values", self.q_values),
            tf.summary.histogram("predictions", self.best_q)
        ])


while True:
    try:
        # Run the network itself
        start_epsilon = 1  # Start exploring with this probability
        end_epsilon = 0.1  # Finish exploring on this probability
        decay_steps = 300  # How many steps epsilon should be decayed from s to end
        batch_size = 10
        update_frequency = 30  # Update target q network towards online dqn
        discount = .99  # Discount for target Q values
        total_episodes = 1000  # Upper bound on number of episodes
        savepath = "./ddqn_offline_model"  # Where to save the model
        out_size = 512  # Size of conv4 that will be splitted to a and v
        update_rate = 0.01  # Update target q network towards online with this rate

        # Temp folder, needed for transferring state and action between Java agent and Python agent
        path_live = "./Temps/"

        # Properties
        tf.reset_default_graph()
        online_QN = DDQN(out_size)
        target_QN = DDQN(out_size)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainable_variables = tf.trainable_variables()
        targetParams = updateTargetTF(trainable_variables, update_rate)
        memory = ExperienceReplay()

        # Where we save our checkpoints and graphs
        summary_dir = os.path.abspath("./ddqn_summaries/{ddqn_offline_model}")
        summary_writer = SummaryStorage(scope="summary", dir=summary_dir)
        epsilon = start_epsilon
        decay_step = (start_epsilon - end_epsilon) / decay_steps
        total_t = 0
        state_maker = StateMaker()

        '''
        if not os.path.exists(path):
            os.makedirs(path)
        '''

        with tf.Session() as sess:
            sess.run(init)
            '''
            if (len(os.listdir(savepath)) > 0):
                print('Loading Model...')
                checkpoint = tf.train.get_checkpoint_state(savepath)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            '''
            # Load memory
            #memory.loadsample()
            #print(len(memory.memory))
            #memory.load()
            print('DONE')
            s = 'None'
            level = 0  # start from first level
            clevel = 4
            last_reward = 0
            for i in range(1, total_episodes):
                if s == 'None':
                    # Get first observation
                    print(os.listdir(path_live))
                    while (len(os.listdir(path_live)) == 0):
                        time.sleep(1)  # wait for something to appear
                    while (len(os.listdir(path_live)[0].split('_')) < 2):
                        time.sleep(1)  # wait for state
                    while (len(sorted(os.listdir(path_live), key=lambda x: int(x.split('_')[0].split('+')[1]))) < 1):
                        time.sleep(2)  # wait for agent to store the current state

                    '''
                    We'll also save the trjactory
                    '''
                    print(os.listdir(path_live))
                    filename = sorted(os.listdir(path_live), key=lambda x: int(x.split('_')[0].split('+')[1]))[0]

                    cfilename = filename
                    print('Done waiting for current state')
                    s = imageio.imread("./Temps/" + filename)[:, :, :3]
                    s = state_maker.make(sess, s)
                    # Delete the state after it was read
                    time.sleep(1)
                    os.remove("./Temps/" + filename)
                    time.sleep(5)

                d = False
                # Greedy choice
                print('epxilon: ', epsilon)
                if np.random.rand(1) < epsilon:
                    # Explore
                    print("Explore a new actions")
                    a = np.random.randint(0, 91)
                    loss = 'None'
                    predicted_q_values = 'None'
                else:
                    # Exploit
                    print("Exploit using NN")
                    print(sess.run(online_QN.best_q, feed_dict={online_QN.imageIn: [s]}))
                    a = sess.run(online_QN.best_q, feed_dict={online_QN.imageIn: [s]})[0]

                # Save action for Java agent
                # only if not done!
                if len(cfilename.split('_')) != 3:
                    file = open("./Temps/action.txt", "w")
                    file.write(str(a))
                    file.close()

                time.sleep(6)

                # Wait for Java agent to save next state and reward that was obtained
                while (len(os.listdir(path_live)) == 0):
                    time.sleep(1)  # wait for something to appear
                while (len(os.listdir(path_live)[0].split('_')) < 2):
                    time.sleep(1)  # wait for state

                while (len(sorted(os.listdir(path_live), key=lambda x: int(x.split('_')[0].split('+')[1]))) < 1):
                    time.sleep(2)  # wait for agent to store the current state

                next_state_name = sorted(os.listdir(path_live), key=lambda x: int(x.split('_')[0].split('+')[1]))[0]
                print('JAVA SENT BACK', os.listdir(path_live))
                print('Done waiting for next state')

                r = int(next_state_name.split('_')[1].split('+')[1].split('.')[0])
                # Uncomment if you would like to use Reward Clippinging technique
                # if (r > 3000):
                #     r = 1
                # else:
                #     r = -1
                r = round(r / levels2[clevel], 2)  # reward is the ratio from total desired score

                s1 = imageio.imread("./Temps/" + next_state_name)[:, :, :3]
                s1 = state_maker.make(sess, s1)

                d = 0
                if len(next_state_name.split('_')) == 3:
                    d = 1
                    level += 1
                    last_reward = 0
                    if next_state_name.split('_')[2] == "Won":
                        r *= 1
                    # else:
                    # r *= -1
                    if level > 20:
                        level = 0

                cfilename = next_state_name
                time.sleep(1)
                os.remove("./Temps/" + next_state_name)

                total_t += 1

                # Decay epsilon
                if epsilon > end_epsilon:
                    epsilon -= decay_step

                s = s1

                if d:
                    s = 'None'  # once done level get new first state

                if s == 'None':
                    pass
                else:
                    if last_reward == 0:
                        memory.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save observation to memory
                    else:
                        memory.remember(np.reshape(np.array([s, a, r - last_reward, s1, d]), [1, 5]))
                    last_reward = r
                    # Save the model every 10 steps
                    # if i % 10 == 0:
                    #    saver.save(sess, savepath + '/model-' + str(i) + '.ckpt')
                    #    print("Saved Model")

                if (total_t % update_frequency) == 0:
                    # batched updates, collect

                    memory.savesample()

                    print("Start training!!!")
                    batch_size = int(len(memory.memory)/4)
                    train_batch = memory.sample(batch_size)

                    for i in range(20):
                    # Feed next state to online qn
                        print(i)
                        Q_online_best = sess.run(online_QN.best_q,
                                                 feed_dict={online_QN.imageIn: np.reshape(np.vstack(train_batch[:, 3]),
                                                                                          [-1, 84, 84, 3])})

                        # Feed next state to offline qn
                        Q_offline = sess.run(target_QN.q_values,
                                             feed_dict={target_QN.imageIn: np.reshape(np.vstack(train_batch[:, 3]),
                                                                                      [-1, 84, 84, 3])})

                        # is end? 0 : 1
                        was_end = -(train_batch[:, 4] - 1)

                        # Evaluate decision of online network using offline network. ----
                        # Double Q learning update: y = R_t+1 + discount * Q(S_t+1, argmax Q(S_t+1,a,online_params), offline_params)

                        # Get Q(S_t+1, argmax Q(S_t+1,a,online_params), offline_params),
                        # by selecting the best q values predicted by online network from offline network
                        double_Q = Q_offline[range(batch_size), Q_online_best]

                        # Update target qs with train batch rewards + discounted target QN q values
                        target_Q = train_batch[:, 2] + (discount * double_Q * was_end)

                        # Feed train batch, update the online network with target q

                        _, summaries = sess.run([online_QN.optimized, online_QN.summaries],
                                                feed_dict={online_QN.imageIn: np.reshape(np.vstack(train_batch[:, 0]),
                                                                                         [-1, 84, 84, 3]),
                                                           online_QN.target_q: target_Q,
                                                           online_QN.actions: train_batch[:, 1]})
                        summary_writer.summary_writer.add_summary(summaries, i)

                        updateTarget(targetParams, sess)  # Update the target qn to online qn with some update rate

                        # Store the summaries
                        episode_summary = tf.Summary()
                        episode_summary.value.add(simple_value=r, tag="reward")
                        episode_summary.value.add(simple_value=a, tag="action")
                        episode_summary.value.add(simple_value=epsilon, tag="epsilon")
                        summary_writer.summary_writer.add_summary(episode_summary, i)
                        summary_writer.summary_writer.flush()

                    s = s1

                    if d:
                        s = 'None'  # once done level get new first state
                    if s == 'None':
                        pass
                    '''
                    else:
                        if last_reward == 0:
                            memory.remember(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save observation to memory
                        else:
                            memory.remember(np.reshape(np.array([s, a, r - last_reward, s1, d]), [1, 5]))
                        last_reward = r
                    '''
                    # Save the model every 10 steps

                    saver.save(sess, savepath + '/model-' + str(i) + '.ckpt')
                    print("Saved Model")

                    print("End of training!!!")
                else:
                    s = s1
                    if d:
                        s = 'None'  # once done level get new first state

    except Exception as e:
        #print("Error: ", e)
        traceback.print_exc()
    finally:
        time.sleep(10)
