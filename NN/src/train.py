from NN.src.storage import save_weights
def train(ann,Input,Target,weights_file="weights",maxIterations = 100000,minError =1e-4,minErrorDelta=1e-6):
    error = 0
    loss = [error]
    for i in range(maxIterations + 1):
        error= ann.backwardProp(Input, Target)
        if i % 1000 == 0:
            print("Iteration {0}\tError: {1:0.8f}".format(i,error))
        if error <= minError:
            print("Minimum error reached at iteration {0}".format(i))
            break
        if abs(error-loss[-1])<=minErrorDelta:
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
def train_batch(ann,Input,Target):
    error = ann.backwardProp(Input, Target)
    return error
