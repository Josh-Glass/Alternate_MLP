from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import model
import itertools
from matplotlib import cm
from sklearn.metrics import accuracy_score

'''
ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

#plot the 3D surface

ax.plot_surface(X, Y, Z, edgecolor= 'royalblue', lw=0.5, rstride= 8, cstride=8, alpha = 0.3)

#plot projection of the contours for each dimension 
#By choosing offsets that match the appropriate axes limits, the projected contours will sit 
#on the 'Walls' of the graph

ax.contourf(X, Y, Z, zdir= 'z', offset= -100, cmap ='coolwarm')
ax.contourf(X, Y, Z, zdir= 'x', offset= -40, cmap ='coolwarm')
ax.contourf(X, Y, Z, zdir= 'y', offset= 40, cmap ='coolwarm')

ax.set(xlim=(-40,40), ylim=(-40,40), zlim=(-100,100),
xlabel='X', ylabel='Y', zlabel='Z')

#plt.show()
'''

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
    },}



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

hps = {
    'learning_rate': .1,  # <-- learning rate
    'weight_range': [6.0, 6.1],  # <-- weight range
    'num_hidden_nodes': 5,

    'hidden_activation': np.sin, # <-- sine function
    'hidden_activation_deriv': np.cos, # <-- sine derivative

    'output_activation': lambda x: x, # <-- linear activation function
    'output_activation_deriv': lambda x: 1, # <-- linear derivative

    # 'output_activation': lambda x: sigmoid(x), 
    # 'output_activation_deriv': lambda x: sigmoid_derivative(x), 
}

nEpochs = 1000

l = np.linspace(0,1,50)
s = np.linspace(0,1,50)

params = model.build_params(
    2,  # <-- num features
    hps['num_hidden_nodes'],
    1, # <-- number of classes
    weight_range = hps['weight_range']
)

num_training_epochs = nEpochs


params = model.fit(params, data['alternate']['inputs'], (data['alternate']['labels'] - .5) * 2, hps, training_epochs = num_training_epochs)
preds=np.round(model.forward(params=params,inputs=data['alternate']['inputs'],hps=hps)[-1])
print(preds)
print((data['alternate']['labels'] - .5) * 2)
print(accuracy_score((data['alternate']['labels'] - .5) * 2,preds))


data['alternate']['probs'] = model.forward(params, np.array([l,s]).T, hps)[-1]

data['alternate']['learned_hid_weights'] = params['input']['hidden']['weights']
data['alternate']['learned_hid_bias'] = params['input']['hidden']['bias']

#scale the probs to be between 0 and 1 
data['alternate']['probs'] -= data['alternate']['probs'].min() #make minimun zero

data['alternate']['probs'] /= data['alternate']['probs'].max() #makes max one

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

prob= data['alternate']['probs'] 






for i in range(data['alternate']['inputs'].shape[0]):
    ax.text(
        data['alternate']['inputs'][i,0],
        data['alternate']['inputs'][i,1],
        data['alternate']['labels'][i,0],
        data['alternate']['txt_labels'][i],
        c = data['alternate']['txt_colors'][i],
        fontsize = 15,
        fontweight = 'bold',
    )



l,s = np.meshgrid(l,s)

surf = ax.plot_surface(s,l,prob,cmap=cm.coolwarm, linewidth=0,antialiased=False)
cbar =fig.colorbar(surf,shrink=0.5, aspect=8)
cbar.set_label('Response Probability', rotation=270,labelpad=15)
ax.set(xlim=(0,1), ylim=(0,1), zlim=(0,1),
xlabel='Angle', ylabel='Size', zlabel='Category Label')
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Category Label',fontsize=15, rotation=270)
ax.zaxis._axinfo["grid"].update({"linewidth":3})
ax.xaxis._axinfo["grid"].update({"linewidth":3})
ax.yaxis._axinfo["grid"].update({"linewidth":3})

ax.set_zticks([0, 1])


ax.view_init(26,-78)
plt.show()
#plt.savefig('3d_WORKS.png')