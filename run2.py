'''
Script to run simulation
'''
# external modules
import numpy as np 

# internal modules
import model

np.random.seed(0)


data = {
    'alternate': {
        'inputs': np.array([
            [.1,.4],
            [.1,.5],
            [.1,.6],
            [.1,.7],


            [.15,.4],
            [.15,.5],
            [.15,.6],
            [.15,.7],

            [.2,.4],
            [.2,.5],
            [.2,.6],
            [.2,.7],




            [.3,.2],
            [.3,.3],
            [.3,.4],
            [.3,.5],

            [.35,.2],
            [.35,.3],
            [.35,.4],
            [.35,.5],


            [.4,.2],
            [.4,.3],
            [.4,.4],
            [.4,.5],



            
            [.5,.4],
            [.5,.5],
            [.5,.6],
            [.5,.7],


            [.55,.4],
            [.55,.5],
            [.55,.6],
            [.55,.7],


            [.6,.4],
            [.6,.5],
            [.6,.6],
            [.6,.7],

            [.7,.2],
            [.7,.3],
            [.7,.4],
            [.7,.5],

            [.75,.2],
            [.75,.3],
            [.75,.4],
            [.75,.5],

            [.8,.2],
            [.8,.3],
            [.8,.4],
            [.8,.5],

        ]),
        'gens': np.array([
            [.9,.2],
            [.9,.3],
            [.9,.4],
            [.9,.5],
            [.9,.6],
            [.9,.7],

            [.95,.2],
            [.95,.3],
            [.95,.4],
            [.95,.5],
            [.95,.6],
            [.95,.7],

            [1.0,.2],
            [1.0,.3],
            [1.0,.4],
            [1.0,.5],
            [1.0,.6],
            [1.0,.7],
        ]),
        'labels': np.array([
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0]]),
        'txt_labels': [
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',],
        'txt_colors': ['orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','orange','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue'],
        'learned_hid_weights': None,
        'learned_hid_bias': None,
        'probs': None,
    },

    'unirule': {
        'inputs': np.array([
            [.1,.4],
            [.1,.5],
            [.1,.6],
            [.1,.7],


            [.15,.4],
            [.15,.5],
            [.15,.6],
            [.15,.7],

            [.2,.4],
            [.2,.5],
            [.2,.6],
            [.2,.7],




            [.3,.2],
            [.3,.3],
            [.3,.4],
            [.3,.5],

            [.35,.2],
            [.35,.3],
            [.35,.4],
            [.35,.5],


            [.4,.2],
            [.4,.3],
            [.4,.4],
            [.4,.5],



            
            [.5,.4],
            [.5,.5],
            [.5,.6],
            [.5,.7],


            [.55,.4],
            [.55,.5],
            [.55,.6],
            [.55,.7],


            [.6,.4],
            [.6,.5],
            [.6,.6],
            [.6,.7],

            [.7,.2],
            [.7,.3],
            [.7,.4],
            [.7,.5],

            [.75,.2],
            [.75,.3],
            [.75,.4],
            [.75,.5],

            [.8,.2],
            [.8,.3],
            [.8,.4],
            [.8,.5]
        ]),
        'gens': np.array([
            [.9,.2],
            [.9,.3],
            [.9,.4],
            [.9,.5],
            [.9,.6],
            [.9,.7],

            [.95,.2],
            [.95,.3],
            [.95,.4],
            [.95,.5],
            [.95,.6],
            [.95,.7],

            [1.0,.2],
            [1.0,.3],
            [1.0,.4],
            [1.0,.5],
            [1.0,.6],
            [1.0,.7],
        ]),
        'labels': np.array([
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [1],[1],[1],[1],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0],
        [0],[0],[0],[0]]),
        'txt_labels': [
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B'],
        'txt_colors': [
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue'],
        'learned_hid_weights': None,
        'learned_hid_bias': None,
        'probs': None,
    },

    'sandwitch': {
        'inputs': np.array([
            [.1,.4],
            [.1,.5],
            [.1,.6],
            [.1,.7],


            [.15,.4],
            [.15,.5],
            [.15,.6],
            [.15,.7],

            [.2,.4],
            [.2,.5],
            [.2,.6],
            [.2,.7],




            [.3,.2],
            [.3,.3],
            [.3,.4],
            [.3,.5],

            [.35,.2],
            [.35,.3],
            [.35,.4],
            [.35,.5],


            [.4,.2],
            [.4,.3],
            [.4,.4],
            [.4,.5],



            
            [.5,.4],
            [.5,.5],
            [.5,.6],
            [.5,.7],


            [.55,.4],
            [.55,.5],
            [.55,.6],
            [.55,.7],


            [.6,.4],
            [.6,.5],
            [.6,.6],
            [.6,.7],

            [.7,.2],
            [.7,.3],
            [.7,.4],
            [.7,.5],

            [.75,.2],
            [.75,.3],
            [.75,.4],
            [.75,.5],

            [.8,.2],
            [.8,.3],
            [.8,.4],
            [.8,.5],        
        ]),
        'gens': np.array([
            [.9,.2],
            [.9,.3],
            [.9,.4],
            [.9,.5],
            [.9,.6],
            [.9,.7],

            [.95,.2],
            [.95,.3],
            [.95,.4],
            [.95,.5],
            [.95,.6],
            [.95,.7],

            [1.0,.2],
            [1.0,.3],
            [1.0,.4],
            [1.0,.5],
            [1.0,.6],
            [1.0,.7],
        ]),
        'labels': np.array([
            [1],[1],[1],[1],
            [1],[1],[1],[1],
            [1],[1],[1],[1],
            [0],[0],[0],[0],
            [0],[0],[0],[0],
            [0],[0],[0],[0],
            [0],[0],[0],[0],
            [0],[0],[0],[0],
            [0],[0],[0],[0],
            [1],[1],[1],[1],
            [1],[1],[1],[1],
            [1],[1],[1],[1]]),
        'txt_labels': [
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'B','B','B','B',
            'A','A','A','A',
            'A','A','A','A',
            'A','A','A','A'],
        'txt_colors': [
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'blue','blue','blue','blue',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',
            'orange','orange','orange','orange',],
        'learned_hid_weights': None,
        'learned_hid_bias': None,
        'probs': None,
    },
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

hps = {
    'learning_rate': .1,  # <-- learning rate
    'weight_range': [6, 6.1],  # <-- weight range
    'num_hidden_nodes': 1,

    'hidden_activation': np.sin, # <-- sine function
    'hidden_activation_deriv': np.cos, # <-- sine derivative

    'output_activation': lambda x: x, # <-- linear activation function
    'output_activation_deriv': lambda x: 1, # <-- linear derivative

    # 'output_activation': lambda x: sigmoid(x), 
    # 'output_activation_deriv': lambda x: sigmoid_derivative(x), 
}

nEpochs = 200



l = np.linspace(0,1,50)
s = np.linspace(0,1,50)
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(
    3,1,
    figsize = [10,6], sharex = True, sharey = True
)



## __ STRUCTURE 1
params = model.build_params(
    2,  # <-- num features
    hps['num_hidden_nodes'],
    1, # <-- number of classes
    weight_range = hps['weight_range']
)

num_training_epochs = nEpochs


params = model.fit(params, data['alternate']['inputs'], (data['alternate']['labels'] - .5) * 2, hps, training_epochs = num_training_epochs)

data['alternate']['probs'] = model.forward(params, np.array([l,s]).T, hps)[-1]

data['alternate']['learned_hid_weights'] = params['input']['hidden']['weights']
data['alternate']['learned_hid_bias'] = params['input']['hidden']['bias']

#scale the probs to be between 0 and 1 
data['alternate']['probs'] -= data['alternate']['probs'].min() #make minimun zero

data['alternate']['probs'] /= data['alternate']['probs'].max() #makes max one

ax[0].plot(
    l,
    (data['alternate']['probs'] - .5) * 2,
    # *np.sin(data['alternate']['learned_hid_weights'] * l + data['alternate']['learned_hid_bias']),
    c = 'red',
)

for i in range(data['alternate']['inputs'].shape[0]):
    ax[0].text(
        data['alternate']['inputs'][i,0],
        (data['alternate']['labels'][i,0] - .5) * 2 + ((data['alternate']['labels'][i,0] - .5) * 2) * .8,
        data['alternate']['txt_labels'][i],
        c = data['alternate']['txt_colors'][i],
        fontsize = 15,
        fontweight = 'bold',
    )




exit()
## __ STRUCTURE 2
params = model.build_params(
    2,  # <-- num features
    hps['num_hidden_nodes'],
    1, # <-- number of classes
    weight_range = hps['weight_range']
)

num_training_epochs = nEpochs
params = model.fit(params, data['unirule']['inputs'], (data['unirule']['labels'] - .5) * 2, hps, training_epochs = num_training_epochs)

data['unirule']['probs'] = model.forward(params, np.array([l,s]).T, hps)[-1]


data['unirule']['learned_hid_weights'] = params['input']['hidden']['weights']
data['unirule']['learned_hid_bias'] = params['input']['hidden']['bias']

data['unirule']['probs'] -= data['unirule']['probs'].min()
data['unirule']['probs'] /= data['unirule']['probs'].max()

ax[1].plot(
    l,
    (data['unirule']['probs'] - .5) * 2,
    # *np.sin(data['unirule']['learned_hid_weights'] * l + data['unirule']['learned_hid_bias']),
    c = 'red',
)

for i in range(data['unirule']['inputs'].shape[0]):
    ax[1].text(
        data['unirule']['inputs'][i,0],
        (data['unirule']['labels'][i,0] - .5) * 2 + ((data['unirule']['labels'][i,0] - .5) * 2) * .8,
        data['unirule']['txt_labels'][i],
        c = data['unirule']['txt_colors'][i],
        fontsize = 15,
        fontweight = 'bold',
    )




## __ STRUCTURE 3
params = model.build_params(
    2,  # <-- num features
    hps['num_hidden_nodes'],
    1, # <-- number of classes
    weight_range = hps['weight_range']
)

num_training_epochs = nEpochs
params = model.fit(params, data['sandwitch']['inputs'], (data['sandwitch']['labels'] - .5) * 2, hps, training_epochs = num_training_epochs)
data['sandwitch']['probs'] = model.forward(params, np.array([l,s]).T, hps)[-1]

data['sandwitch']['learned_hid_weights'] = params['input']['hidden']['weights']
data['sandwitch']['learned_hid_bias'] = params['input']['hidden']['bias']

data['sandwitch']['probs'] -= data['sandwitch']['probs'].min()
data['sandwitch']['probs'] /= data['sandwitch']['probs'].max()

ax[2].plot(
    l,
    (data['sandwitch']['probs'] - .5) * 2,
    # *np.sin(data['sandwitch']['learned_hid_weights'] * l + data['sandwitch']['learned_hid_bias']),
    c = 'red',
)

for i in range(data['sandwitch']['inputs'].shape[0]):
    ax[2].text(
        data['sandwitch']['inputs'][i,0],
        (data['sandwitch']['labels'][i,0] - .5) * 2 + ((data['sandwitch']['labels'][i,0] - .5) * 2) * .8,
        data['sandwitch']['txt_labels'][i],
        c = data['sandwitch']['txt_colors'][i],
        fontsize = 15,
        fontweight = 'bold',
    )





fig.text(
    0.04, 0.5, 
    'Response Probability', ha='center', va='center', rotation='vertical',
    fontsize = 20, fontweight = 'bold',
)


ax[0].set_yticks([-1,1]); ax[0].set_yticklabels(['B','A'], fontsize = 13, fontweight = 'bold')
ax[1].set_yticks([-1,1]); ax[1].set_yticklabels(['B','A'], fontsize = 13, fontweight = 'bold')
ax[2].set_yticks([-1,1]); ax[2].set_yticklabels(['B','A'], fontsize = 13, fontweight = 'bold')

ax[0].set_ylim([-2,3])
ax[1].set_ylim([-2,3])
ax[2].set_ylim([-2,3])

ax[0].axhline(.0, linestyle = '--', color = 'red', alpha = .3)
ax[1].axhline(.0, linestyle = '--', color = 'red', alpha = .3)
ax[2].axhline(.0, linestyle = '--', color = 'red', alpha = .3)




# ax[0].imshow(
#     np.array([data['alternate']['probs'][:,0]]) ,
#     extent = [*ax[0].get_xlim(), *ax[0].get_ylim()],
#     aspect = 'auto', alpha = .8, vmin = -1, vmax = 1,
#     cmap = 'binary',
# )

# ax[1].imshow(
#     np.array([data['unirule']['probs'][:,0]]) ,
#     extent = [*ax[1].get_xlim(), *ax[1].get_ylim()],
#     aspect = 'auto', alpha = .8, vmin = -1, vmax = 1,
#     cmap = 'binary',
# )

# ax[2].imshow(
#     np.array([data['sandwitch']['probs'][:,0]]) ,
#     extent = [*ax[2].get_xlim(), *ax[2].get_ylim()],
#     aspect = 'auto', alpha = .8, vmin = -1, vmax = 1,
#     cmap = 'binary',
# )


# plt.tight_layout()
# plt.suptitle('Response Probabilities of an MLP', fontsize = 20, fontweight = 'bold')
plt.savefig('figure_2d.png')
# plt.savefig('figure.eps')

