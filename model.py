'''
MultiLayer Classifier (aka, multilayer perceptron)
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

--- Functions ---
    - forward <-- get model outputs
    - loss <-- cost function
    - loss_grad <-- returns gradients
    - response <-- luce-choice rule (ie, softmax without exponentiation)
    - fit <-- trains model on a number of epochs
    - predict <-- gets class predictions
    - build_params <-- returns dictionary of weights
    - update_params <-- updates weights


--- Notes ---
    - implements sum-squared-error cost function
    - hidden activation function & derivative have to be provided in 'hps' dictionary
'''
## external requirements
import numpy as np 


## "forward pass"
def forward(params, inputs, hps):
    hidden_act_raw = np.add(
        np.matmul(
            inputs,
            params['input']['hidden']['weights']
        ),
        params['input']['hidden']['bias']
    )

    hidden_act = hps['hidden_activation'](hidden_act_raw)

    output_act_raw = np.add(
        np.matmul(
            hidden_act,
            params['hidden']['output']['weights']
        ),
        params['hidden']['output']['bias'],
    )

    output_act = hps['output_activation'](output_act_raw)

    return [hidden_act_raw, hidden_act, output_act_raw, output_act]


## cost function (mean squared error)
def loss(params, inputs, targets, hps):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs, hps)[-1],
                targets
            )
        )
    ) / inputs.shape[0]


## backprop (for sum squared error cost function)
# loss_grad = grad(loss) # <-- this is how easy it would have been if we used autograd instead
def loss_grad(params, inputs, targets, hps): # <-- backprop in numpy
    hidden_act_raw, hidden_act, output_act_raw, output_act = forward(params, inputs, hps)

    ## gradients for decode layer ( chain rule on cost function )
    decode_grad = np.multiply(
        hps['output_activation_deriv'](output_act_raw),
        (2 * (output_act - targets))  / inputs.shape[0] # <-- deriv of cost function
    )

    ## gradients for decode weights
    decode_grad_w = np.matmul(
        hidden_act.T,
        decode_grad
    )

    ## gradients for decode bias
    decode_grad_b = np.matmul(
        [[1]],
        decode_grad
    )

    # - - - - - - - - - -

    ## gradients for encode layer ( chain rule on hidden layer )
    encode_grad = np.multiply(
        hps['hidden_activation_deriv'](hidden_act_raw),
        np.matmul(
            decode_grad, 
            params['hidden']['output']['weights'].T
        )
    )

    ## gradients for encode weights
    encode_grad_w = np.matmul(
        inputs.T,
        encode_grad
    )

    ## gradients for encode bias
    encode_grad_b = np.matmul(
        [[1]],
        encode_grad
    )

    return {
        'input': {
            'hidden': {
                'weights': encode_grad_w,
                'bias': encode_grad_b,
            }
        },
        'hidden': {
            'output': {
                'weights': decode_grad_w,
                'bias': decode_grad_b,
            }
        }
    }


## luce choice
def response(params, inputs, hps):
    return utils.softmax(
        forward(params, inputs, hps)[-1]
    )

## build parameter dictionary
def build_params(num_features, num_hidden_nodes, num_classes, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_classes <-- number of categories in the dataset
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.zeros([1, num_hidden_nodes]),
            }
        },
        'hidden': {
            'output': {
                'weights': np.ones([num_hidden_nodes, num_classes]),
                'bias': np.ones([1, num_classes]),
            }
        }
    }


## weight update
def update_params(params, gradients, lr):
    for layer in params:
        for connection in gradients[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


## fit to training set
def fit(params, inputs, targets, hps, training_epochs = 1, randomize_presentation = True):
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)
        
        for i in range(inputs.shape[0]):

            params = update_params(
                params, 
                loss_grad(params, inputs[i:i+1,:], targets[i:i+1,:], hps), # <-- returns gradients
                hps['learning_rate'],
            )

    return params

## predict
def predict(params, inputs, hps):
    return np.argmax(
        forward(params, inputs, hps)[-1],
        axis = 1
    )