import numpy as np
import math


def exp_dot(w, f):
    """
    :param w: array-like, 1*n
    :param f: array-like, 1*n
    :return: float
    """
    return math.exp(np.dot(w, f))


class CrfInput(object):
    def __init__(self, x, y, label_num, features_fun, predict=False):
        self.x = x
        self.y = y
        self.start_y = -1    # 初始状态
        self.labels = range(label_num)
        self.label_num = label_num
        self.features_fun = features_fun  # 特征函数集合
        self.predict = predict    # 是否是预测
        self.all_features_by_x = []    # [{j:{i:feats}}]
        self.features_by_xy = []    # [[]]
        self.F = []    # 各特征值在序列上的求和， [], 1*len(features)
        self.f_sharp = 1    # F#的值, 所有特征的和
        self.get_x_features()  # 获取x所有情况下的特征
        if not self.predict:
            self.get_xy_features()  # 获取xy的特征
            self.F = np.sum(self.features_by_xy, 0)
            self.f_sharp = np.sum(self.F)

    def get_x_features(self):
        """基于x获取所有可能的特征"""
        for i, xi in enumerate(self.x):
            tem_fs = {}    # {j:{i:feats}}
            yis = self.labels
            if i == 0:
                yis = [self.start_y]
            for yi in yis:
                tem_fs_yi = {}    # {i:feats}
                for yj in self.labels:
                    tem_fs_yi[yj] = [f(x, yi, yj, i) for f in self.features_fun]
                tem_fs[yi] = tem_fs_yi
            self.all_features_by_x.append(tem_fs)

    def get_xy_features(self):
        """基于xy，获取当前样本特征"""
        for i, xi in enumerate(self.x):
            if i == 0:
                tem_fs = [f(x, self.start_y, self.y[i], i) for f in self.features_fun]
            else:
                tem_fs = [f(x, self.y[i-1], self.y[i], i) for f in self.features_fun]
            self.features_by_xy.append(tem_fs)


class CrfCal(object):
    def __init__(self, crf_input: CrfInput, w):
        self.crf_input = crf_input
        self.start_y = -1    # 初始状态
        self.w = w    # 特征权重
        self.z_w = 1    # 分母
        self.alphas = []
        self.betas = []
        self.cal()

    def cal(self):
        self.alpha()
        self.beta()

    def alpha(self):
        """
        前向计算
        :param fs: 特征，[{j:{i:feats}}]
        :return:z_w, alphas:[t[i]]
        """
        for t, tem_f in enumerate(self.crf_input.all_features_by_x):
            if t == 0:
                alpha = [exp_dot(self.w, tem_f[-1][i]) for i in self.crf_input.labels]
            else:
                alpha = []
                for i in self.crf_input.labels:
                    new_alpha_i = sum([self.alphas[t-1][j] * exp_dot(self.w, tem_f[j][i]) for j in self.crf_input.labels])
                    alpha.append(new_alpha_i)
            self.alphas.append(alpha)
        self.z_w = sum(self.alphas[-1])

    def beta(self):
        """
        后向计算
        :param fs: 特征，[{j:{i:feats}}]
        :return:z_w, betas:[t[i]]
        """
        self.betas = [[0] for i in self.crf_input.all_features_by_x]
        T = len(self.crf_input.x)
        for t in reversed(range(T)):
            if t == T - 1:
                beta = [1 for i in self.crf_input.labels]
            else:
                tem_f = self.crf_input.all_features_by_x[t + 1]
                beta = []
                for i in self.crf_input.labels:
                    new_beta_i = sum([exp_dot(self.w, tem_f[i][j]) * self.betas[t+1][j] for j in self.crf_input.labels])
                    beta.append(new_beta_i)
            self.betas[t] = beta
        # todo beta计算z_w，书上有误
        z_w = sum([exp_dot(self.w, self.crf_input.all_features_by_x[0][-1][j]) * self.betas[0][j] for j in self.crf_input.labels])
        return z_w

    def p_yi_t(self, t):
        yi = self.crf_input.y[t]
        p = self.alphas[t][yi] * self.betas[t][yi]/self.z_w
        return p

    def p_yij_t(self, t):
        yi = self.start_y if t == 0 else self.crf_input.y[t - 1]
        yj = self.crf_input.y[t]
        if t == 0:
            p = exp_dot(self.w, self.crf_input.all_features_by_x[t][yi][yj]) * self.betas[t][yj] / self.z_w
        else:
            p = self.alphas[t-1][yi] * exp_dot(self.w, self.crf_input.all_features_by_x[t][yi][yj]) * self.betas[t][yj] / self.z_w
        return p

    def f_yx(self):
        """特征在P(x,y)下的期望"""
        # todo 书上公式错误？
        ef = [0 for i in self.crf_input.features_fun]
        for t, ft in enumerate(self.crf_input.features_by_xy):
            fs_xy_t = self.crf_input.features_by_xy[t]
            for k, fk in enumerate(fs_xy_t):
                diff = fk * self.p_yij_t(t)
                ef[k] += diff
        return ef

    def f_yx_d(self, k, d):
        """d*特征k在P(x,y)下的期望"""
        ef = 0
        for t, ft in enumerate(self.crf_input.features_by_xy):
            fs_xy_t = self.crf_input.features_by_xy[t]
            diff = self.p_yij_t(t) * fs_xy_t[k] * d
            ef += diff
        return ef

    def w_f_yx_hat(self):
        """w*f在 \hat P(x,y)下的期望, BFGS loss的一部分"""
        wf_hat = [0 for i in self.crf_input.features_fun]
        for t, fs_xy_t in enumerate(self.crf_input.features_by_xy):
            for k, fk in enumerate(fs_xy_t):
                diff = np.dot(self.w, fk)
                wf_hat[k] += diff
        return wf_hat

    def pw(self):
        """crf概率"""
        pw = exp_dot(self.w, self.crf_input.F) / self.z_w
        return pw

    def f_yx_pw(self):
        """BFGS 梯度的一部分"""
        pw = self.pw()
        f_pw = [0 for i in self.crf_input.features_fun]
        for t, fs_xy_t in enumerate(self.crf_input.features_by_xy):
            for k, fk in enumerate(fs_xy_t):
                diff = pw * fk
                f_pw[k] += diff
        return f_pw


class Crf(object):
    def __init__(self, xs, ys, label_num, features_fun):
        self.xs = xs
        self.ys = ys
        self.labels = range(label_num)
        self.label_num = label_num
        self.features_fun = features_fun    # 特征函数集合
        self.start_label = -1
        self.w = [0 for i in self.features_fun]
        self.crf_inputs = [CrfInput(x, y, self.label_num, self.features_fun) for x, y in zip(self.xs, self.ys)]

    def e_f_yx_hat(self):
        Fs = [crf_input.F for crf_input in self.crf_inputs]
        ef = np.mean(Fs, 0)
        return ef

    def predict(self, x):
        """维特比预测算法"""
        deltas = []
        phis = []
        result_y = [1 for i in x]
        crf_input = CrfInput(x, result_y, self.label_num, self.features_fun, True)
        for i, xi in enumerate(x):
            delta, phi = [0 for i in self.labels], {}
            if i == 0:
                for yi in self.labels:
                    delta[yi] = np.dot(self.w, crf_input.all_features_by_x[i][self.start_label][yi])
                    phi[yi] = 0
            else:
                for yi in self.labels:
                    tem_delta = [deltas[i-1][yj] + np.dot(self.w, crf_input.all_features_by_x[i][yj][yi]) for yj in self.labels]
                    max_index = int(np.argmax(tem_delta))
                    delta[yi] = tem_delta[max_index]
                    phi[yi] = max_index
            deltas.append(delta)
            phis.append(phi)
        y_t = int(np.argmax(deltas[-1]))
        result_y[-1] = y_t
        for t in reversed(range(len(x)-1)):
            y_t1 = result_y[t+1]
            result_y[t] = phis[t+1][y_t1]
        return result_y

    def newton_method(self, fx, gx):
        """
        牛顿法求fx=0的解
        :param fx: 方程
        :param gx: 梯度
        :return:
        """
        x = 0
        while True:
            fx_i = fx(x)
            if abs(fx_i) < 0.001:
                break
            gx_i = gx(x)
            x = x - fx_i/gx_i
        return x

    def iis(self):
        """改进的迭代尺度学习算法"""
        ef_hat = self.e_f_yx_hat()
        f_sharps = [crf_input.f_sharp for crf_input in self.crf_inputs]
        done = False
        cycle = 0
        while not done:
            cycle += 1
            done = True
            crf_cals = [CrfCal(crf_input, self.w) for crf_input in self.crf_inputs]
            print(self.loss(self.w))
            # efs = [crf_cal.f_yx() for crf_cal in crf_cals]
            pws = [crf_cal.pw() for crf_cal in crf_cals]
            Fs = [crf_input.F for crf_input in self.crf_inputs]
            for k in range(len(self.features_fun)):
                # f_dk = lambda dk: np.mean([crf_cal.f_yx_d(k, np.exp(dk*f_sharp))
                #     for crf_cal, f_sharp in zip(crf_cals, f_sharps)], 0) - ef_hat[k]
                # g_dk = lambda dk: np.mean([crf_cal.f_yx_d(k, np.exp(dk*f_sharp) * f_sharp)
                #     for crf_cal, f_sharp in zip(crf_cals, f_sharps)], 0)
                # dk = self.newton_method(f_dk, g_dk)
                # dk2 = ef_hat[k]/(np.mean([ef[k] for ef in efs], 0))
                # dk2 /= 7
                # print(dk, dk2)

                # todo 书上公式错误？上面注释部分为书上公式

                f_dk = lambda dk: np.mean([pw * F[k] * np.exp(dk * f_sharp)
                                       for pw, F, f_sharp in zip(pws, Fs, f_sharps)], 0) - ef_hat[k]
                g_dk = lambda dk: np.mean([pw * F[k] * np.exp(dk * f_sharp) * f_sharp
                                       for pw, F, f_sharp in zip(pws, Fs, f_sharps)], 0)
                dk = self.newton_method(f_dk, g_dk)
                # dk2 = ef_hat[k]/(sum([pw * np.mean([tf[k] for tf in fs_xy]) for pw, fs_xy in zip(pws, fs_xys)], 0))
                # dk2 /= 7
                # print(dk, dk2)

                self.w[k] += dk
                if abs(dk) > 0.0001:
                    done = False
            if cycle > 160:
                break

    def loss(self, w):
        """BFGS loss函数"""
        n = len(self.xs)
        crf_cals = [CrfCal(crf_input, w) for crf_input in self.crf_inputs]
        loss = (np.sum([math.log(crf_cal.z_w) for crf_cal in crf_cals])
                -np.sum([crf_cal.w_f_yx_hat() for crf_cal in crf_cals]))/n
        return loss

    def gradient(self, w):
        """BFGS loss的梯度函数"""
        n = len(self.xs)
        crf_cals = [CrfCal(crf_input, w) for crf_input in self.crf_inputs]
        g = (np.sum([crf_cal.f_yx_pw() for crf_cal in crf_cals], 0)
             - np.sum([crf_input.F for crf_input in self.crf_inputs], 0)) / n
        return g

    def bfgs(self):
        """拟牛顿法BFGS学习算法"""
        loss_w = self.loss
        g_w = self.gradient
        I = np.identity(len(self.w))
        D = np.identity(len(self.w))
        g_k = g_w(self.w)
        cycle = 0
        while True:
            cycle += 1
            print(self.loss(self.w))
            if np.linalg.norm(g_k) < 0.0001:
                return
            pk = -np.dot(D, g_k)
            lambda_k = self.linear_search(loss_w, pk)
            delta = lambda_k * pk
            self.w += delta
            if cycle > 20:
                return
            g_k_1 = g_w(self.w)
            if np.linalg.norm(g_k_1) < 0.0001:
                return
            # 更新D
            y = g_k_1 - g_k
            n = np.dot(delta, y)    # 分母
            m = np.outer(delta, y)    # 分子
            D = np.dot(
                np.dot(I-m/n, D),
                np.transpose(I - m/n)
                )+np.outer(delta, delta)/n
            g_k = g_k_1

    def linear_search(self, fx, x):
        """
        tree.py有类似实现，非凸时不一定最优
        黄金分割搜索法，假设区间在0.00001与0.1之间
        """
        a = 0.00001
        b = 0.1
        e = 0.00001
        while b - a > e:
            loss_a1 = fx(self.w + a * x)
            loss_b1 = fx(self.w + b * x)
            if loss_a1 < loss_b1:
                b = a * 0.382 + b * 0.618
            else:
                a = a * 0.618 + b * 0.382
        return (a + b)/2


if __name__ == '__main__':
    x = "我爱北京天安门"
    y = [0, 0, 0, 1,0, 1, 1]
    features = []
    # for xi in x:
    features.append(lambda a, b, c, d: 1 if a[d]=="我" and c==0 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="爱" and c==0 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="北" and c==0 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="京" and c==1 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="天" and c==0 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="安" and c==1 else 0)
    features.append(lambda a, b, c, d: 1 if a[d]=="门" and c==1 else 0)
    w = [0 for i in features]
    crf_input = CrfInput(x, y, 2, features)
    crf_cal = CrfCal(crf_input, w)
    print(crf_cal.pw())
    print(crf_cal.z_w)
    print(crf_cal.beta())
    xs = [x]
    ys = [y]
    xs.append("北京我爱你")
    ys.append([0, 1, 0, 0, 0])
    xs.append("我爱天安门")
    ys.append([0, 0, 0, 1, 1])
    crf = Crf(xs, ys, 2, features)
    # crf.iis()
    crf.bfgs()
    print(crf.w)
    crf_cal = CrfCal(crf_input, crf.w)
    print(crf_cal.pw())
    print(crf_cal.z_w)
    print(crf.predict(x))