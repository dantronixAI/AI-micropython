from .numpy_package import numpy as np
class rmsprop():
    def __init__(self,
                lr = 0.01,
                alpha = 0.99,
                eps = 1e-08,
                weight_decay = 0,
                momentum = 0,
                centered = False):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
    def get_layer(self):
        return  rmsprop_layer(
                lr = self.lr,
                alpha = self.alpha,
                eps = self.eps,
                weight_decay = self.weight_decay,
                momentum = self.momentum,
                centered = self.centered)


class rmsprop_layer():
    def __init__(self,
                lr,
                alpha,
                eps,
                weight_decay,
                momentum,
                centered):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

        self.vt_old = 0
        self.bt_old = 0
        self.gt_old = 0
        self.theta_old = 0

        self.vt_now = 0
        self.bt_now = 0
        self.gt_now = 0
        self.theta_now = 0

    def optimize(self,grad):
        gt = grad
        if self.weight_decay!=0:
            gt=gt+self.weight_decay*self.theta_old
        self.vt_now=self.alpha*self.vt_old+(1 - self.alpha)*gt*gt
        vt_aux = self.vt_now
        if self.centered:
            self.gt_now=self.gt_old*self.alpha+(1 - self.alpha)*gt
            vt_aux = vt_aux - (self.gt_now)*(self.gt_now)
        if self.momentum>0:
            self.bt_now = self.momentum*self.bt_old + gt/(np.sqrt(vt_aux)+self.eps)
            self.theta_now = self.theta_old - self.lr*self.bt_now
        else:
            self.theta_now = self.theta_old - self.lr * gt/(np.sqrt(vt_aux)+self.eps)

        self.vt_old = self.vt_now
        self.bt_old = self.bt_now
        self.gt_old = self.gt_now
        self.theta_old = self.theta_now

        return self.theta_now


class sgd():
    def __init__(self,
                lr=0.1,
                weight_decay=0,
                momentum=0,
                dampening=0,
                nesterov=False,
                maximize=False):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
    def get_layer(self):
        return  sgd_layer(
                lr = self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                dampening=self.dampening,
                nesterov=self.nesterov,
                maximize=self.maximize)

class sgd_layer():
    def __init__(self,
                lr,
                weight_decay,
                momentum,
                dampening,
                nesterov,
                maximize):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize

        self.bt_old = 0
        self.gt_old = 0
        self.theta_old = 0

        self.bt_now = 0
        self.gt_now = 0
        self.theta_now = 0

        self.t = 0

    def optimize(self,grad):
        gt = grad
        if self.weight_decay!=0:
            gt=gt+self.weight_decay*self.theta_old
        if self.momentum!=0:
            if self.t>1:
                self.bt_now = self.momentum*self.bt_old+(1-self.dampening)*gt
            else:
                self.bt_now = grad
                self.t+=1
            if self.nesterov:
                gt = self.gt_old+self.momentum*self.bt_now
            else:
                gt = self.bt_now
        if self.maximize:
            self.theta_now = self.theta_old + self.lr*gt
        else:
            self.theta_now = self.theta_old - self.lr * gt

        self.bt_old = self.bt_now
        self.gt_old = self.gt_now
        self.theta_old = self.theta_now
        return self.theta_now



class adagrad():
    def __init__(self,
                lr=0.01,
                lr_decay=0,
                weight_decay=0,
                initial_accumulator_value=0,
                eps=1e-10):
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps =eps
    def get_layer(self):
        return  adagrad_layer(
                lr = self.lr,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
                initial_accumulator_value=self.initial_accumulator_value,
                eps=self.eps)


class adagrad_layer():
    def __init__(self,
                lr,
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps):
        self.lr = lr
        self.lr_decay= lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

        self.theta_old = 0
        self.state_sum_old = 0

        self.theta_now = 0
        self.state_sum_now = 0
        self.t = 1

    def optimize(self,grad):
        gt = grad
        lr_aux = self.lr/(1.0+(self.t-1.0)*self.lr_decay)
        if self.weight_decay !=0:
            gt = gt + self.weight_decay*self.theta_old
        self.state_sum_now=self.state_sum_old+gt*gt
        self.theta_now = self.theta_old - lr_aux*gt/(np.sqrt(self.state_sum_now)+self.eps)

        self.state_sum_old = self.state_sum_now
        self.theta_old = self.theta_now
        self.t+=1
        return self.theta_now



class adam():
    def __init__(self,
                 lr=0.001,
                 betas=(0.9,0.999),
                 weight_decay=0,
                 eps=1e-8,
                 amsgrad=False,
                 maximize=False):
        self.lr=lr
        self.betas=betas
        self.weight_decay=weight_decay
        self.eps=eps
        self.amsgrad=amsgrad
        self.maximize=maximize

    def get_layer(self):
        return  adam_layer(
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
            maximize=self.maximize)

class adam_layer():
    def __init__(self,
                lr,
                betas,
                weight_decay,
                eps,
                amsgrad,
                maximize):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.B1 = betas[0]
        self.B2 = betas[1]

        self.vt_max = 0

        self.vt_old = 0
        self.mt_old = 0
        self.theta_old = 0

        self.vt_now = 0
        self.mt_now = 0
        self.theta_now = 0
        self.t = 1

    def optimize(self,grad):
        if self.maximize:
            gt = -grad
        else:
            gt = grad

        if self.weight_decay!=0:
            gt = gt +self.weight_decay*self.theta_old

        self.mt_now = self.B1 * self.mt_old + (1 - self.B1) * gt
        self.vt_now = self.B2 * self.vt_old + (1 - self.B2) * gt * gt

        mt_ = self.mt_now/(1-(self.B1**self.t))
        vt_ = self.vt_now/(1-(self.B2**self.t))

        if self.amsgrad:
            self.vt_max = max(self.vt_max,vt_)
            self.theta_now = self.theta_old - self.lr * mt_/(np.sqrt(self.vt_max)+self.eps)
        else:
            self.theta_now = self.theta_old - self.lr * mt_ / (np.sqrt(vt_) + self.eps)

        self.mt_old = self.mt_now
        self.vt_old = self.vt_now
        self.theta_old = self.theta_now
        self.t+=1
        return self.theta_now

class adamw():
    def __init__(self,
                 lr=0.001,
                 betas=(0.9,0.999),
                 weight_decay=0.01,
                 eps=1e-8,
                 amsgrad=False,
                 maximize=False):
        self.lr=lr
        self.betas=betas
        self.weight_decay=weight_decay
        self.eps=eps
        self.amsgrad=amsgrad
        self.maximize=maximize

    def get_layer(self):
        return  adamw_layer(
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
            maximize=self.maximize)

class adamw_layer():
    def __init__(self,
                lr,
                betas,
                weight_decay,
                eps,
                amsgrad,
                maximize):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.B1 = betas[0]
        self.B2 = betas[1]

        self.vt_max = 0

        self.vt_old = 0
        self.mt_old = 0
        self.theta_old = 0

        self.vt_now = 0
        self.mt_now = 0
        self.theta_now = 0
        self.t = 1

    def optimize(self,grad):
        if self.maximize:
            gt = -grad
        else:
            gt = grad

        self.theta_now = self.theta_old- self.weight_decay*self.lr*self.theta_old
        self.mt_now = self.B1 * self.mt_old + (1 - self.B1) * gt
        self.vt_now = self.B2 * self.vt_old + (1 - self.B2) * gt * gt

        mt_ = self.mt_now/(1-(self.B1**self.t))
        vt_ = self.vt_now/(1-(self.B2**self.t))

        if self.amsgrad:
            self.vt_max = max(self.vt_max,vt_)
            self.theta_now = self.theta_old - self.lr * mt_/(np.sqrt(self.vt_max)+self.eps)
        else:
            self.theta_now = self.theta_old - self.lr * mt_ / (np.sqrt(vt_) + self.eps)

        self.mt_old = self.mt_now
        self.vt_old = self.vt_now
        self.theta_old = self.theta_now
        self.t+=1
        return self.theta_now

class radam():
    def __init__(self,
                 lr=0.001,
                 betas=(0.9,0.999),
                 weight_decay=0.01,
                 eps=1e-8):
        self.lr=lr
        self.betas=betas
        self.weight_decay=weight_decay
        self.eps=eps

    def get_layer(self):
        return  radam_layer(
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps)

class radam_layer():
    def __init__(self,
                lr,
                betas,
                weight_decay,
                eps):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

        self.B1 = betas[0]
        self.B2 = betas[1]

        self.vt_max = 0

        self.vt_old = 0
        self.mt_old = 0
        self.theta_old = 0

        self.vt_now = 0
        self.mt_now = 0
        self.theta_now = 0
        self.p_inf = (2/(1-self.B2))-1
        self.t = 1

    def optimize(self,grad):
        gt = grad
        if self.weight_decay!=0:
            gt = gt +self.weight_decay*self.theta_old

        self.mt_now = self.B1 * self.mt_old + (1 - self.B1) * gt
        self.vt_now = self.B2 * self.vt_old + (1 - self.B2) * gt * gt

        mt_ = self.mt_now/(1-(self.B1**self.t))
        B2_p = (self.B2**self.t)
        pt = self.p_inf - 2*self.t*B2_p/(1-B2_p)

        if pt>5:
            lt = np.sqrt((1-B2_p)/(self.vt_now+self.eps))
            rt = np.sqrt((pt-4)*(pt-2)*self.p_inf/((self.p_inf-4)*(self.p_inf-2)*pt))
            self.theta_now = self.theta_old - self.lr * mt_*rt*lt
        else:
            self.theta_now = self.theta_old - self.lr * mt_

        self.mt_old = self.mt_now
        self.vt_old = self.vt_now

        self.theta_old = self.theta_now
        self.t+=1
        return self.theta_now