# 创建时间： 2024/2/29 14:29
from utils import *
def disp_img(X, filename):
    from utils import normalize, imgrid
    import matplotlib.pyplot as plt
    # im = imgrid(normalize(X), 10, 10, 56, 46, 2)
    # im = imgrid(normalize(X), 10, 15, 55, 40, 2)
    im = imgrid(normalize(X), 10, 10, 60, 40, 2)
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight')

def test_PCA():
    from scipy.io import loadmat
    from L2PCA import L2PCA
    from utils import compute_reconstruction_performance
    from datetime import datetime

    # filename = './data/ATNTFace400Image56x46scale_noise_reg_nonoverlap.mat'
    filename = 'Face_GT_10x10_60x40_diming.mat'
    filename_m = './data/matlab.mat'
    # data = loadmat(filename)['Dn']

    # # ATnT
    # clean_data = loadmat(filename)['DIDX']
    # x0 = clean_data[:, 0:100]
    # # print(x0)
    # disp_img(x0, './PCA_results/x0.png')

    # data = loadmat(filename)['D']
    # X = data[:, 0:150]
    # disp_img(X, './PCA_results/X.png') # 输入图片
    clean_data = loadmat(filename)['D']
    x0 = clean_data[:, 0:100]
    # print(x0)
    disp_img(x0, './PCA_results/x0.png')

    data = loadmat(filename)['Dn']
    X = data[:, 0:100]
    print(X)
    disp_img(X, './PCA_results/x0.png')

    data_ = loadmat(filename_m)['A_hat']
    X_ = data_[:, 0:100]
    print('X_',X_)


    file = './PCA_results/' + 'R_PCA'


    compute_reconstruction_performance(x0, X, X_, True)
    # compute_AR_reconstruction_performance(X, X_, True)
    disp_img(X_, './PCA_results/X_RPCA.png')

    # kmeans
    # kmeans
    # kmeans
    label = loadmat(filename)['L']
    y = label[:, 0:150]
    y = y.T
    y = y.reshape(-1)
    y = y - 60
    # print("y", y)
    x, y_true, num_class = read_data(X_)
    # print('x.shape', x.shape)

    num_trails = 10

    s_C = np.zeros([num_class, num_class, num_trails], dtype=np.int32)
    s_C_opt = np.zeros([num_class, num_class, num_trails], dtype=np.int32)
    s_loss = np.zeros([num_trails, ])
    s_acc = np.zeros([num_trails, ])
    s_acc_opt = np.zeros([num_trails, ])

    for i in range(num_trails):
        loss, y_pred, C, acc = run_kmeans(x.T, y_true, num_class)  # X: n-by-p
        C_opt, acc_opt = reorder_confusion_matrix(C)
        s_C[:, :, i] = C
        s_C_opt[:, :, i] = C_opt
        s_loss[i] = loss
        s_acc[i] = acc
        s_acc_opt[i] = acc_opt
        print('%d-th trial: loss=%f' % (i + 1, loss))

    index = np.argmin(s_loss)
    print('Pick %d-th trail with lowest objective function value' % (index + 1))
    C = s_C[:, :, index]
    acc = s_acc[index]
    C_opt = s_C_opt[:, :, index]
    acc_opt = s_acc_opt[index]
    print('Original accuracy: %f' % acc)
    print('After reordering, optimal accuracy: %f' % acc_opt)
    # file = './PCA_results/' + clfname
    file_writer = open(file + 'acc.txt', 'w')
    file_writer_content = ""
    file_writer_content += ('%.2f' % (acc))
    file_writer_content += '\n'
    file_writer.write(file_writer_content)
    file_writer.close()

    file_writer = open(file + 'acc_opt.txt', 'w')
    file_writer_content = ""
    file_writer_content += ('%.2f' % (acc_opt))
    file_writer_content += '\n'
    file_writer.write(file_writer_content)
    file_writer.close()


def main():
    test_PCA()

if __name__ == '__main__':
    main()