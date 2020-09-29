import os
import psutil

import torch
import matplotlib.pyplot as plt

# patternnet
import layers as patlayers


def get_model_easy(model):
    # remove layers/check for layers which are not Cov2d, BatchNorm2d, MaxPool2d or relu
    model_easy = [] # model without Dropout and identity layer
    for layer in model: 
        if layer.__class__.__name__ == 'Dropout':
            continue
        elif (layer.__class__.__name__ == 'Expression' and 
              layer.expression_fn.__name__ == 'identity'):
            continue
        model_easy.append(layer)
        if not (layer.__class__.__name__ in ['Conv2d', 'BatchNorm2d', 'MaxPool2d'] or 
                (layer.__class__.__name__ == 'Expression' and 
                 layer.expression_fn.__name__ == 'relu')):
            print(layer)
    return model_easy


# forward_pass_for_statistics
def forward_pass_for_statistics(model, pattern_layers, inp):
    out = model[1](model[0](inp))
    conv_in_outs = []
    for layer in pattern_layers:
        layerclass = layer.__class__.__name__
        if layerclass == 'PatternConv2d':
            layer_in_out = [out]
            out = layer.forward(out) 
            if out.__class__.__name__ == 'tuple':
                layer_in_out.extend(out) # Extends list by appending elements from the iterable. give [1,2,3,4,5] instead of [1,2,3,[4,5]]
                out = out[0]
            else:
                layer_in_out.append(out)
           # print(*[elem.shape for elem in layer_in_out])
            conv_in_outs.append(layer_in_out) # Appends object at the end.
        elif layerclass == 'PatternMaxPool2d':
            out, _ = layer.forward(out) 
        elif layerclass == 'PatternReLU':
            out, _ = layer.forward(out) 
        elif layerclass == 'PatternBatchNorm2d':
            out = layer.forward(out)
        else:
            out = layer(out)
         #   print('Layer class not known:', layerclass)
    out = model[-1](model[-2](out))
    return out, conv_in_outs


# define forward_pass_for signal
def forward_pass_for_signal(model, pattern_layers, inp):
    out = model[1](model[0](inp))
    pool_sizes = []
    pool_inds = []
    relu_inds = []
    for layer in pattern_layers:
        layerclass = layer.__class__.__name__
        if layerclass == 'PatternConv2d':
            out = layer.forward(out) # Performs the forward computation of the layer it was initialized on and returns the layer’s regular output and the output without the bias added. 
            if out.__class__.__name__ == 'tuple':
                out = out[0]
        elif layerclass == 'PatternMaxPool2d':
            pool_sizes.append(out.shape)
            out, inds = layer.forward(out) #it returns the indices of the maximum locations of the max pooling operation so that these switches can be used for the backward computation. 
            pool_inds.append(inds)
        elif layerclass == 'PatternReLU':
            out, inds = layer.forward(out) #it returns the indices of the units that were smaller or equal to zero. These indices are needed when calling the layer’s backward method to set the same units to zero as in the forward pass. 
            relu_inds.append(inds)
        elif layerclass == 'PatternBatchNorm2d':
            out = layer.forward(out)
#     out = model[-2:](out)
    return out, pool_inds, relu_inds, pool_sizes


# original forward function
def original_forward(model, inp):
    for layer in model:
        inp = layer(inp)
        if layer.__class__.__name__ == 'Dropout' or (layer.__class__.__name__ == 'Expression' and layer.expression_fn.__name__ == 'identity'):
            continue
    return inp


# define backward pass
def backward_pass(pattern_layers, out, pool_inds, relu_inds, pool_sizes):
    pool_cnt = 0
    relu_cnt = 0
    # make a copy of out
    signal = torch.tensor(out)
    # go through the layers in backward order
    for layer in pattern_layers[::-1]:
        layerclass = layer.__class__.__name__
        if layerclass == 'PatternConv2d':
        #    print("Shape before conv:", signal.shape)
            signal = layer.backward(signal)
         #   print('Shape after conv:', signal.shape)
        elif layerclass == 'PatternMaxPool2d':
        #    print("Shape before maxunpool:", signal.shape)
            signal = layer.backward(signal, pool_inds[pool_cnt], pool_sizes[pool_cnt])
            pool_cnt += 1
         #   print('Shape after maxunpool:', signal.shape)
        elif layerclass == 'PatternReLU':
            signal = layer.backward(signal, relu_inds[relu_cnt])
            relu_cnt += 1
        elif layerclass == 'PatternBatchNorm2d':
            signal = layer.backward(signal)
    return signal


def create_pattern_layers(model_easy):
    pattern_layers = []
    for layer in model_easy:
        layerclass = layer.__class__.__name__
        if layerclass == 'Conv2d':
            pattern_layers.append(patlayers.PatternConv2d(layer))
        elif layerclass == 'MaxPool2d':
            pattern_layers.append(patlayers.PatternMaxPool2d(layer))
        elif layerclass == 'Expression' and layer.expression_fn.__name__ == 'relu':
            pattern_layers.append(patlayers.PatternReLU())
        elif layerclass == 'BatchNorm2d':
            pattern_layers.append(patlayers.PatternBatchNorm2d(layer))
        elif layerclass == 'Expression' and layer.expression_fn.__name__ == 'identity':
            pattern_layers.append(layer)
        else:
            print('No PatternLayer for layer:', layer)
    return pattern_layers


def ram_usage():
    process = psutil.Process(os.getpid())
    usage = process.memory_info().rss
    return usage/1e9


def plot_data_and_signal(data, signal):
    plt.figure(figsize=(20,15))

    for i in range(signal.shape[3]):
        plt.subplot(signal.shape[3],1,i+1)
        plt.plot(signal[0,0,:,i].detach().cpu().detach().numpy(), label='signal')
        plt.plot(data[0,i,:,0].cpu().detach().numpy(), label='eeg')

    plt.ylim(signal.cpu().detach().numpy().min(), signal.cpu().detach().numpy().max())
    plt.legend()
    plt.show()
    
    
def plot_spectrum_for_windows(windows, sfreq):
    data = windows.get_data()
    # calculate spectrum usind rfft
    # calculate fft using rfft for real data
    # The function rfft calculates the FFT of a real sequence and outputs
    # the complex FFT coefficients for only half of the frequency range. 
    # using abs value, because we want the real part
    # getting the artificial part using angle
    # 
    spectrum = np.abs(np.fft.rfft(data, axis=2))
    # This function computes the one-dimensional n-point discrete Fourier Transform (DFT) 
    # of a real-valued array by means of an efficient algorithm called 
    # the Fast Fourier Transform (FFT).
    # calculate mean across events
    mean_spectrum = np.mean(spectrum, axis=0)
    # calculate frequency
    freqs = np.fft.rfftfreq(data.shape[-1], 1/sfreq)
    return mean_spectrum, freqs, spectrum


