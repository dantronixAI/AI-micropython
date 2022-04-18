from NN.src.storage import save_weights
verbose_train =100
verbose_train_batch = 10
minIterations = 10
def train(ann,Input,Target,
          weights_file="weights",
          maxIterations = 100000,
          minError =1e-4,
          minErrorDelta=1e-6):
    error = 0
    loss = [error]
    print("Beginning Training")
    for i in range(maxIterations + 1):
        error= ann.backwardProp(Input, Target)
        if ((i % verbose_train == 0) & (i!=0)):
            print("Iteration {0}\tError: {1:0.8f}".format(i,error))
        if ((error <= minError)& (i>minIterations)):
            print("Minimum error reached at iteration {0}".format(i))
            break
        if ((abs(error - loss[-1]) <= minErrorDelta) & (i>minIterations)):
            print("Minimum delta error reached at iteration {0}".format(i))
            break
        loss.append(error)
    loss = loss[1:]
    if i == maxIterations:
        print("Maximum iterations reached : {}".format(maxIterations) )
    print("Error: {0:0.8f}".format(error))
    print("Done Training")
    save_weights(ann,weights_file)
    return loss



def train_batch(ann,Input,Target,
                batch_size=10,
                weights_file='weigths',
                maxIterations=100000,
                minError=1e-4,
                minErrorDelta=1e-6
                ):

    size_sample = len(Input)
    error = 0
    loss = [error]
    for i in range(maxIterations + 1):
        ini_pos = 0
        end_pos = batch_size
        error_cumulative = 0
        while end_pos < size_sample:
            error_cumulative += ann.backwardProp(Input[ini_pos:end_pos], Target[ini_pos:end_pos])

            ini_pos = end_pos
            end_pos = end_pos + batch_size
            #  Include last batch
            if ((ini_pos < size_sample - 1) & (end_pos > size_sample)):
                end_pos = size_sample - 1
        error =error_cumulative
        if ((i % verbose_train_batch == 0) & (i!=0)):
            print("Iteration {0}\tError: {1:0.8f}".format(i, error))
        if ((error <= minError)& (i>minIterations)):
            print("Minimum error reached at iteration {0}".format(i))
            break
        if ((abs(error - loss[-1]) <= minErrorDelta) & (i>minIterations)):
            print("Minimum delta error reached at iteration {0}".format(i))
            break
        loss.append(error)
    loss = loss[1:]
    if i == maxIterations:
        print("Maximum iterations reached : {}".format(maxIterations) )
    print("Error: {0:0.8f}".format(error))
    print("Done Training")
    save_weights(ann,weights_file)
    return loss
