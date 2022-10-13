import pandas as pd

def plot_accuracy(history, fname=None, **kwargs):
    '''
    Plots the history of accuracy and validation accuracy from the history object
    returned by a Keras model fitting. If you pass a string to `fname`, the plot will
    also be saved at that location.    

    Note that any keyword arguments other than fname are simply passed on to pd.DataFrame.plot()
    For example, passing `title = 'title string'` will add a title to the plot. 
    '''
    plt = pd.DataFrame({"Accuracy":history.history['accuracy'], 
                         "Validation accuracy":history.history['val_accuracy']}).plot(xlabel = 'Epoch', **kwargs)
    if fname is not None:
        fig = plt.get_figure()
        fig.savefig(fname)
    return plt

