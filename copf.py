import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam, stopGradientOrNot,\
    regParams, params, l2_normalize
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time
import random


class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler
        self.n_fold = 100
        self.best_epoch = 0
        self.best_HR = 0.0
        self.best_NDCG = 0.0
        self.save_flag = False
        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNum, args.latdim // 2])
        if args.data == 'beibei':
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR45', 'NDCG45', 'HR50', 'NDCG50', 'HR55', 'NDCG55', 'HR60',
                    'NDCG60', 'HR65', 'NDCG65', 'HR100', 'NDCG100']
        else:
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35',
                    'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        if name == 'Test':
            if self.best_HR <= reses['HR']:
                self.best_HR = round(reses['HR'], 4)
                self.best_NDCG = round(reses['NDCG'], 4)
                self.best_epoch = ep
                self.save_flag = True
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        train_time = 0
        test_time = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch()
            t1 = time()
            train_time += t1 - t0
            print('Train_time', t1 - t0, 'Total_time', train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test and (ep > 90 or ep < 2):
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3 - t2
                print('Test_time', t3 - t2, 'Total_time', test_time)
                log(self.makePrint('Test', ep, reses, test))
            # if self.save_flag:
            #     self.saveHistory()
            #     self.save_flag = False
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))

        log('----------Best Performance----------')
        log('Epoch %d/%d, Test: ' % (self.best_epoch, args.epoch) + 'HR = %.4f, ' % (self.best_HR) + 'NDCG = %.4f' % (
            self.best_NDCG))

        # ADD
        log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        # log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')
        log_file = open(
            log_dir + '/alllog', 'a')

        log_file.write(
            "gnn_layer: " + str(args.gnn_layer) + " gnn_mtl_layer: " + str(args.gnn_mtl_layer) + " encoder:" + str(args.gnn) +
            " decoder:" + str(args.decoder) + "\n")
        log_file.write(
            'Epoch %d/%d, Test: ' % (self.best_epoch, args.epoch) + 'HR = %.4f, ' % (self.best_HR) + 'NDCG = %.4f' % (
                self.best_NDCG))
        # log_file.write(self.makePrint('Test', args.epoch, reses, True))
        log_file.write("\n")
        log_file.write("\n")

        # if self.save_flag:
        #     self.saveHistory()
        #     self.save_flag = False

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (args.user + args.item) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = args.user + args.item
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def mess_drop(self, embs):
        return tf.nn.dropout(embs, 1 - self.mess_dropout[0])


    def defineModel(self, allEmbed):
        if args.data == 'tmall':
            all_trnMats = [self.handler.trnMats[0], self.handler.two_trnMats, self.handler.three_trnMats, self.handler.all_trnMats]
        else:
            all_trnMats = [self.handler.trnMats[0],self.handler.two_trnMats,self.handler.all_trnMats]
        self.adjs_cas = []
        for trnMats in all_trnMats:
            R = trnMats.tolil()

            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])

            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))

            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)

            if args.normalization == "left":
                self.all_trnMats = left_trn
            elif args.normalization == "right":
                self.all_trnMats = right_trn
            elif args.normalization == "symm":
                self.all_trnMats = symm_trn
            elif args.normalization == 'none':
                self.all_trnMats = adj_mat.tocsr()
            adj = self.all_trnMats
            idx, data, shape = transToLsts(adj, norm=False)
            self.adjs_cas.append(tf.sparse.SparseTensor(idx, data, shape))


        gnn_layer = eval(args.gnn_layer)
        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        ego_embeddings = allEmbed
        for beh in range(args.behNum):
            all_embeddings = [ego_embeddings]
            for index in range(gnn_layer[beh]):
                symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs_cas[beh], all_embeddings[-1])
                if args.encoder == 'lightgcn':
                    lightgcn_embeddings = symm_embeddings
                    lightgcn_embeddings = lightgcn_embeddings + all_embeddings[-1]
                    all_embeddings.append(lightgcn_embeddings)
            ego_embeddings = all_embeddings[-1] + ego_embeddings
            all_embeddings = tf.add_n(all_embeddings)
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [args.user, args.item], 0)
        self.ulat_merge, self.ilat_merge = tf.add_n(self.ulat), tf.add_n(self.ilat)

    def gnn_predict(self, src):
        gnn_layer = eval(args.gnn_mtl_layer)
        uids = self.uids[src]
        iids = self.iids[src]

        tmp_emb_user = self.ulat[src]
        tmp_emb_item = self.ilat[src]

        src_ulat = tf.nn.embedding_lookup(tmp_emb_user, uids)
        src_ilat = tf.nn.embedding_lookup(tmp_emb_item, iids)

        metalat111 = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.behNum, reg=True, useBias=True,
                        activation='softmax', name='gate111', reuse=True)
        w1 = tf.reshape(metalat111, [-1, args.behNum, 1])
        exper_info = [src_ulat * src_ilat]

        for index in range(args.behNum):
            if index != src:

                ego_embeddings = tf.concat([(self.ulat[index] * args.index + self.ulat[src] * args.src) / 2,
                                            (self.ilat[index] * args.index + self.ilat[src] * args.src) / 2], axis=0)
                all_embeddings = [ego_embeddings]
                beh_embeddings = tf.expand_dims(self.behEmbeds[src], axis=0)

                for index in range(gnn_layer[src]):
                    if index > 0:
                        beh_embeddings = FC(beh_embeddings, args.latdim // 2, reg=True, useBias=False,
                                            activation=None, name='layer_' + str(index), reuse=True)

                    symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[src], all_embeddings[-1] * beh_embeddings)
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = symm_embeddings + all_embeddings[-1]

                        all_embeddings.append(lightgcn_embeddings)

                all_embeddings = tf.add_n(all_embeddings)
                if args.stop_gradient:
                    index_ulat, index_ilat = stopGradientOrNot(index, all_embeddings)
                else:
                    index_ulat, index_ilat = tf.split(all_embeddings, [args.user, args.item], 0)

                exper_info.append(
                    tf.nn.embedding_lookup(index_ulat, uids) * tf.nn.embedding_lookup(index_ilat, iids))

        predEmbed = tf.stack(exper_info, axis=2)

        gnn_out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = tf.squeeze(tf.reduce_sum(gnn_out, axis=-1))

        return preds * args.mult

    def create_multiple_adj_mat(self, adj_mat):
        def left_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()


    def cal_cl_loss(self, beh1, beh2):
        # uids1 = self.uids[beh1]
        # iids1 = self.iids[beh1]
        uids2 = self.uids[beh2]
        iids2 = self.iids[beh2]

        uids2, _ = tf.split(uids2, num_or_size_splits=2)
        iids2, _ = tf.split(iids2, num_or_size_splits=2)


        if args.ssl_mode in ['user_side', 'both_side']:
            user_emb1 = tf.nn.embedding_lookup(self.ulat[beh1], uids2)
            user_emb2 = tf.nn.embedding_lookup(self.ulat[beh2], uids2)  



            normalize_user_emb1 = l2_normalize(user_emb1, axis=1)
            normalize_user_emb2 = l2_normalize(user_emb2, axis=1)

            normalize_all_user_emb1 = l2_normalize(self.ulat[beh1], axis=1)


            pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2),
                                           axis=1)
            pos_score_user = tf.exp(pos_score_user / args.ssl_temp)


            ttl_score_user = tf.matmul(normalize_user_emb2,
                                       normalize_all_user_emb1, transpose_b=True)
            ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / args.ssl_temp), axis=1) 


            ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        if args.ssl_mode in ['item_side', 'both_side']:
            item_emb1 = tf.nn.embedding_lookup(self.ilat[beh1], iids2)
            item_emb2 = tf.nn.embedding_lookup(self.ilat[beh2], iids2)

            normalize_item_emb1 = l2_normalize(item_emb1, axis=1)
            normalize_item_emb2 = l2_normalize(item_emb2, axis=1)
            normalize_all_item_emb1 = l2_normalize(self.ilat[beh1], axis=1)


            pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
            ttl_score_item = tf.matmul(normalize_item_emb2, normalize_all_item_emb1, transpose_b=True)

            pos_score_item = tf.exp(pos_score_item / args.ssl_temp)
            ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / args.ssl_temp), axis=1)

            ssl_loss_item = -tf.reduce_sum(tf.log(pos_score_item / ttl_score_item))



        if args.ssl_mode == 'user_side':
            ssl_loss = args.ssl_reg * ssl_loss_user * (1 / args.user)
        elif args.ssl_mode == 'item_side':
            ssl_loss = args.ssl_reg * ssl_loss_item * (1 / args.user)
        else:
            ssl_loss = args.ssl_reg * (ssl_loss_user + ssl_loss_item) * (1 / args.user)

        return ssl_loss

    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.uids, self.iids = [], []
        self.uids2, self.iids2 = [], []
        self.iids_other = []
        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []

        for i in range(args.behNum):
            R = self.handler.trnMats[i].tolil()

            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])

            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))

            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats

        for i in range(args.behNum):
            adj = self.final_trnMats[i]
            idx, data, shape = transToLsts(adj, norm=False)
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))

            self.uids.append(tf.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))
            tmp = []
            tmp1 = []
            tmp2 = []
            for j in range(args.behNum):
                if i != j and isinstance(self.handler.trnMats_uni_final[i][j], int) == False:
                    tmp.append(
                        tf.placeholder(name='iids_other' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                    tmp1.append(
                        tf.placeholder(name='uids2' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                    tmp2.append(
                        tf.placeholder(name='iids2' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                else:
                    tmp.append(0)
                    tmp1.append(0)
                    tmp2.append(0)
            self.iids_other.append(tmp)
            self.uids2.append(tmp1)
            self.iids2.append(tmp2)


        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], reg=True)
        allEmbed = tf.concat([uEmbed0, iEmbed0], axis=0)

        if args.gnn == 'cogcn':
            self.defineModel(allEmbed)

        self.preLoss = 0
        self.all_cl_loss = 0
        self.parallel_loss = []
        self.mtl_loss = []
        self.cl_loss = []
        self.coefficient = eval(args.coefficient)
        self.all_cl_loss_gl = 0
        self.cl_loss_gl = []

        if args.decoder == 'dfme':
            for beh in range(args.behNum - 1):
                cl_loss = self.cal_cl_loss(beh, args.behNum - 1)
                self.cl_loss.append(cl_loss)
                self.all_cl_loss += cl_loss

            for src in range(args.behNum):
                preds = self.gnn_predict(src)

                sampNum = tf.shape(self.uids[src])[0] // 2
                posPred = tf.slice(preds, [0], [sampNum])
                negPred = tf.slice(preds, [sampNum], [-1])

                self.mtl_loss.append(self.coefficient[src] * (tf.reduce_mean(tf.nn.softplus(-(posPred - negPred))) * 3))
                self.preLoss += self.mtl_loss[src]
                if src == args.behNum - 1:
                    self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        self.loss = self.preLoss + self.regLoss + self.all_cl_loss 

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=globalStep)
        self.optimizer = tf.train.AdamOptimizer(learningRate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat, labelMat_uni):
        temLabel = labelMat[batIds].toarray()
        temLabel_uni = labelMat_uni[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        iLocs_uni = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            posset_uni = np.reshape(np.argwhere(temLabel_uni[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset), len(posset_uni))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
                poslocs_uni = [np.random.choice(args.item)]
                neglocs_uni = [poslocs_uni[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                poslocs_uni = np.random.choice(posset_uni, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
                neglocs_uni = negSamp(temLabel_uni[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                posloc_uni = poslocs_uni[j]
                negloc = neglocs[j]
                negloc_uni = neglocs_uni[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                iLocs_uni[cur] = posloc_uni
                iLocs_uni[cur + temlen // 2] = negloc_uni
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        iLocs_uni = iLocs_uni[:cur] + iLocs_uni[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs, iLocs_uni

    def sampleTrainBatch_ori(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss, epochClLoss = [0] * 3
        mtlLoss = [0, 0, 0]
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.train_op, self.preLoss, self.regLoss, self.loss, self.mtl_loss, self.all_cl_loss]
            feed_dict = {}
            for beh in range(args.behNum):
                uLocs, iLocs = self.sampleTrainBatch_ori(batIds, self.handler.trnMats[beh])
                trnmat_uni = self.handler.trnMats_uni_final[beh]
                for beh_uni in range(args.behNum):
                    if isinstance(trnmat_uni[beh_uni], int) == True:
                        feed_dict[self.uids[beh]] = uLocs
                        feed_dict[self.iids[beh]] = iLocs
                    else:
                        uLocs2, iLocs2, iLocs_uni = self.sampleTrainBatch(batIds, self.handler.trnMats[beh],
                                                                          trnmat_uni[beh_uni])
                        feed_dict[self.uids[beh]] = uLocs
                        feed_dict[self.iids[beh]] = iLocs


            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            preLoss, regLoss, loss, mtl_loss, cl_loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            epochClLoss += cl_loss
            mtlLoss = [x + y for x, y in zip(mtlLoss, mtl_loss)]
        log(f'mtlLoss:{[x / steps for x in mtlLoss]}')
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        ret['clLoss'] = epochClLoss / steps
        return ret

    def sampleTestBatch_allitem(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = []
        iLocs = []
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            if isinstance(posloc, int) == True or type(posloc) in [np.int64]:
                posloc = int(posloc)
                locset = np.concatenate((negset, np.array([posloc])))
            else:
                locset = np.concatenate((negset, np.array(posloc)))

            tstLocs[i] = locset
            uLocs += [batIds[i]] * len(locset)
            iLocs += list(locset)
        return uLocs, iLocs, temTst, tstLocs

    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch_allitem(batIds, self.handler.trnMats[-1])
            feed_dict[self.uids[-1]] = uLocs
            feed_dict[self.iids[-1]] = iLocs

            preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes_allitem(preds, temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg

        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes_allitem(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        batch = len(tstLocs)
        st = 0
        for j in range(batch):
            u_item_num = len(tstLocs[j]) + st
            cur_pred = preds[st:u_item_num]
            st = u_item_num
            predvals = list(zip(cur_pred, tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if isinstance(temTst[j], int) == True or type(temTst[j]) in [np.int64]:
                temTst[j] = int(temTst[j])
                if temTst[j] in shoot:
                    hit += 1
                    ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
            else:
                for eachTst in temTst[j]:
                    if eachTst in shoot:
                        hit += 1
                        ndcg += np.reciprocal(np.log2(shoot.index(eachTst) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':

    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()) + ' ' + args.gnn + ' ' + args.decoder, 'w')


    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text


    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()
