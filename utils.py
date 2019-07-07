
class GraphEmbedding:
    def __init__(self, vocab_size, sentences, sentences_weight, word2load, short_load,embedding_size=50,
                 is_weighted=True, window_size=5, batch_size=64, K=10, learning_rate=0.01):
        """
        完成word2vec算法，但同时支持使用最短路进行CBOW中向量的加权计算（原本算法中不考虑节点之间的距离信息），
        这里引入最短路来引入距离信息，可以发现无论是训练loss还是辅助其他的异常点检测算法都有提升
        """
        self.vocab_size = vocab_size
        self.sentences = sentences
        self.embedding_size = embedding_size
        self.sentences_weight = sentences_weight
        self.is_weighted = is_weighted
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.K=K
        self.word2load= word2load
        self.short_load = short_load
        self.index_from_count()
        if self.is_weighted:
            self.word_vectors, self.word_weights, self.labels = self.preprocess_sentences()
        else:
            self.word_vectors, self.labels = self.preprocess_sentences()
        print(f'训练集准备完毕！共有{len(self.word_vectors)}条训练集')
        self._init_graph()

    def _init_graph(self):
        """
        建立计算图
        :return:
        """
        self.sess = tf.Session()
        self.context_vector = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_vector')
        if self.is_weighted:
            self.context_weight = tf.placeholder(dtype=tf.float32, shape=[None, None], name='context_weight')
            self.context_weight_norm = tf.div(self.context_weight, tf.reshape(tf.reduce_sum(self.context_weight, axis=1), (-1,1)))
        self.output_vector = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='output_vector')

        self.weight = self._init_weight()
        self.context_vector_v = tf.nn.embedding_lookup(
            self.weight['embedding_v'],
            self.context_vector,
            name='context_vector_v'
        )
        if self.is_weighted:
            self.context_vector_v_average = tf.reduce_sum(
                tf.multiply(self.context_vector_v, tf.expand_dims(self.context_weight_norm,-1)),
                axis=1,
                name='context_vector_v_average'
            )
        else:
            self.context_vector_v_average = tf.reduce_mean(
                self.context_vector_v,
                axis=1,
                name='context_vector_v_average'
            )
        self.output_vector_u = tf.nn.embedding_lookup(
            self.weight['embedding_u'],
            self.output_vector,
            name='output_vector_u'
        )

        biase = tf.Variable(
            tf.zeros(shape=[self.vocab_size])
        )
        self.loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.weight['embedding_u'],
            biases=biase,
            inputs=self.context_vector_v_average,
            labels=self.output_vector,
            num_sampled=self.K,
            num_classes=self.vocab_size,
            name='nce_loss'
        ))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_batch(self, word_vectors, labels, word_weights=None):
        step = int(len(word_vectors)/self.batch_size) + 1
        for i in range(step):
            try:
                word_batch = word_vectors[i*self.batch_size:(i+1)*self.batch_size, :]
                label_batch = labels[i*self.batch_size:(i+1)*self.batch_size, :]
                if self.is_weighted:
                    weight_batch = word_weights[i*self.batch_size:(i+1)*self.batch_size, :]
                    yield word_batch, weight_batch, label_batch
                else:
                    yield word_batch, label_batch
            except:
                word_batch = word_vectors[i * self.batch_size:, :]
                label_batch = labels[i * self.batch_size:, :]
                if self.is_weighted:
                    weight_batch = word_weights[i * self.batch_size:, :]
                    yield word_batch, weight_batch, label_batch
                else:
                    yield word_batch, label_batch
                    
    def fit_on_batch(self, word_batch, label_batch, weight_batch=None):
        if self.is_weighted:
            feed_dict = {
                self.context_vector:word_batch,
                self.output_vector:label_batch,
                self.context_weight:weight_batch
            }
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        else:
            feed_dict = {
                self.context_vector: word_batch,
                self.output_vector: label_batch,
            }
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss.mean()

    def fit(self, word_vectors, labels, word_weights=None, epochs=30):
        fit_loss = []
        for epoch in range(epochs):
            step = int(len(word_vectors) / self.batch_size) + 1
            total_loss = []
            
            perm = np.random.permutation(len(word_vectors))
            if self.is_weighted:
                for word_batch, weight_batch, label_batch in \
                        tqdm_notebook(self.get_batch(word_vectors[perm], labels[perm], word_weights[perm]), total=step, postfix=f'epoch:{epoch}', leave=False):
                    if len(word_batch)==0:
                        continue
                    loss = self.fit_on_batch(word_batch, label_batch, weight_batch)
                    if math.isnan(loss):
                        print(word_batch, label_batch, weight_batch)
                    total_loss.append(loss)
            else:
                for word_batch, label_batch in \
                        tqdm_notebook(self.get_batch(word_vectors[perm], labels[perm]), total=step, postfix=f'epoch:{epoch}'):
                    if len(word_batch)==0:
                        continue
                    loss = self.fit_on_batch(word_batch, label_batch)
                    if math.isnan(loss):
                        print(word_batch, label_batch, weight_batch)
                    total_loss.append(loss)
            print(f'epoch:{epoch} loss:{np.mean(total_loss)}')
            fit_loss.append(np.mean(total_loss))
            self.wordVectors = self.sess.run(self.weight['embedding_u'])
            self.word_norm = np.sqrt(np.square(self.wordVectors).sum(axis=1))
        return fit_loss

    def _init_weight(self):
        """
        初始化权重
        :return:
        """
        weight = dict()
        weight['embedding_u'] = tf.Variable(
            tf.random_normal(shape=[self.vocab_size+1, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_u'
        )
        weight['embedding_v'] = tf.Variable(
            tf.random_normal(shape=[self.vocab_size+1, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_v'
        )
        return weight
    
    def index_from_count(self):
        """
        根据词频倒序排列
        :return:
        """
        self.vocab = dict()
        for sentence in self.sentences:
            if isinstance(sentence, list) and len(set(sentence))!=1:
                for word in sentence:
                    try:
                        self.vocab[word] += 1
                    except:
                        self.vocab[word] = 1
        word_sort_by_count = [item[0] for item in sorted(self.vocab.items(), key=lambda r: r[1], reverse=True) if item[1]>0]
        
        self.word2index = dict(zip(word_sort_by_count, range(1,len(self.vocab)+1)))
        self.word2index['pad'] = 0
        self.index2word = dict([(item[1], item[0]) for item in self.word2index.items()])
        self.vocab_size = len(self.word2index)

    def preprocess_sentences(self):
        """
        处理句子
        :return:
        """
        word_vectors = []
        word_weights = []
        labels = []

        for sentence in self.sentences:
            if isinstance(sentence, list) and len(set(sentence))!=1:
                sentence = ['pad']*self.window_size + sentence + ['pad']*self.window_size

                for word_index in range(self.window_size, len(sentence)-self.window_size):
                    word_vector = [self.word2index[sentence[word_index+i]] for i in range(-self.window_size, self.window_size+1) if i != 0]
                    label = self.word2index[sentence[word_index]]
                    sentences_weight = [0 if self.index2word[i]=='pad' else 1/self.short_load[self.word2load[self.index2word[i]], self.word2load[self.index2word[label]]] for i in word_vector]
                    
                    if self.is_weighted:
                        word_weight = [0 if self.index2word[word_vector[i]]=='pad' else sentences_weight[i] for i in range(2*self.window_size)]
                        word_weights.append(word_weight)

                    word_vectors.append(word_vector)
                    labels.append([label])
        if self.is_weighted:
            return np.array(word_vectors), np.array(word_weights), np.array(labels)
        else:
            return np.array(word_vectors), np.array(labels)
    
    def get_most_similar(self, positive, negative=None, k=10):
        positive_vector = np.zeros(self.embedding_size)
        if positive is None:
            pass
        else:
            for pos in positive:
                positive_vector += self.wordVectors[self.word2index[pos], :]

        negative_vector = np.zeros(self.embedding_size)
        if negative is None:
            pass
        else:
            for neg in negative:
                negative_vector += self.wordVectors[self.word2index[neg], :]
        vec = positive_vector - negative_vector
        vec_norm = np.sqrt(np.square(vec).sum())
        cos_similar = self.wordVectors.dot(vec)/(self.word_norm * vec_norm)
        top_k = np.argsort(cos_similar)[-k-1:-1]
        return top_k
        
def floyd_graph(matrix):
    """
    输入邻接矩阵，生成任意两个节点之间的最短路，使用floyd算法
    思路为依次设定每一个节点为中转站，然后更新最短路，算法复杂度为
    O(N^3)，这里的图可以是有向图也可以是无向图。
    :param matrix:
    :return:
    """
    vertices = len(matrix)

    for mid in tqdm_notebook(range(vertices)):
        for row in range(vertices):
            for col in range(vertices):
                if row == col:
                    continue
                else:
                    if matrix[row][mid] + matrix[mid][col]< matrix[row][col]:
                        matrix[row][col] = matrix[row][mid] + matrix[mid][col]
    return matrix
    
def point_closeness(matrix, type='dangalchev'):
    """
    这里完成击中点集中度的计算方式
    base: 1/sum(d(i,j))，这里d(i,j)指的是两个点之间的距离，这种计算方式的缺陷是只能应用于连通图中。
    latora: sum(1/d(i,j))，来自于Latora和Marchiori的论文：https://arxiv.org/PS_cache/cond-mat/pdf/0101/0101396v2.pdf
    dangalchev: sum(1/(2^(d(i,j))))，和上面的计算方式类似，只不过使用2的负指数幂替代了倒数的方式。来自于Dangalchev
    的论文：Residual closeness in networks。该文认为与上面的计算方式比，这种衡量方式更加直观而简洁。

    此函数的返回值为一个列表，每一个值代表对应点的紧密度。
    :param matrix:
    :param type:
    :return:
    """
    output = []
    vertices = len(matrix)

    if type == 'base':
        for i in tqdm_notebook(range(vertices)):
            distance = 0
            for j in range(vertices):
                if matrix[i][j]==999999:
                    raise Exception('此图为非连通图！base方法不能用于非连通图，请考虑其它计算紧密度的方式。')
                else:
                    distance += matrix[i][j]
            output.append(1/distance)
        return output
    elif type == 'latora':
        for i in tqdm_notebook(range(vertices)):
            close = 0
            for j in range(vertices):
                close += 1/matrix[i][j]
            output.append(close)
        return output
    elif type == 'dangalchev':
        for i in tqdm_notebook(range(vertices)):
            close = 0
            for j in range(vertices):
                close += 1/(2**matrix[i][j])
            output.append(close)
        return output
    else:
        print('没有对应的计算方式')
 def PersonalRank(graph, initial_state, alpha):
    """
    二分图随机游走算法实现
    :param graph: nx图格式
    :param initial_state:
    :param alpha: 阻尼系数
    :return:字典格式，按照概率值从大到小排列
    """
    M = nx.adjacency_matrix(graph)
    M = M/M.sum(axis=1)

    n = M.shape[0]
    A = np.eye(n) - alpha*M.T
    b = (1-alpha) * initial_state
    r = solve(A, b)
    result = dict(zip(graph.node, r.reshape((1, -1)).tolist()[0]))
    return result
    
def generate_similarity(M, max_step, min_eps=0.01):
    """
    根据案件-人物矩阵生成相似度矩阵，论文：A generalized model of relational similarity.pdf
    :param M:
    :return:
    """
    O1 = np.identity(M.shape[0])
    O2 = np.identity(M.shape[1])

    step = 0
    while True:
        ori_O1 = O1
        ori_O2 = O2

        M_actor_mean = M.mean(axis=1)
        M_actor = M-M_actor_mean

        actor_numerator = np.dot(np.dot(M_actor, O2), M_actor.T)
        actor_denominator = np.sqrt(np.dot(np.diag(actor_numerator).T, np.diag(actor_numerator)))
        O1 = actor_numerator/actor_denominator

        M_item_mean = M.mean(axis=0)
        M_item = M-M_item_mean

        item_numerator = np.dot(np.dot(M_item, O1), M_item.T)
        item_denominator = np.sqrt(np.dot(np.diag(item_numerator).T, np.diag(item_numerator)))
        O2 = item_numerator/item_denominator

        step += 1
        eps_O1 = np.square(O1-ori_O1).sum()
        eps_O2 = np.square(O2-ori_O2).sum()

        if step % 5 == 0:
            print(f'第{step}步迭代完成， 迭代误差为O1:{eps_O1}, O2:{eps_O2}')

        if min(eps_O1, eps_O2)<min_eps:
            print(f'迭代已收敛，当前迭代步数为{max_step}')
            break

        if step >= max_step:
            print(f'达到最大步数，当前迭代误差为O1:{eps_O1}, O2:{eps_O2}')
            break
    return O1, O2
  
 def get_regression_score(M, epochs=50):
    """
    得到根据回归得到的异常点得分，这里换用lgb进行训练，每次选择一个x作为目标变量，其余x抽取一半作为
    自变量，然后使用模型预测误差作为异常点得分，循环50次集成增加模型性能和泛化能力。
    :param M: 
    :return: 
    """
    
    #para
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'num_leaves': 32,
    'verbose': -1,
    'max_depth': 5,
    'lambda_l2': 0.01, 'lambda_l1': 4
    }
    score=np.zeros(M.shape[0])
    drop_index = []
    
    for epoch in tqdm_notebook(range(epochs)):
        for i in range(M.shape[1]):
            select_index = [m for m in range(M.shape[0]) if m not in drop_index]
            
            M_ori = M[select_index,:]
            
            y = M_ori[:, i]
            x = M_ori[:, [j for j in range(M.shape[1]) if j != i]]
            model = lgb.train(params=params,train_set=lgb.Dataset(x,y))
            y_pred = model.predict(M[:,[j for j in range(M.shape[1]) if j != i]])
            score_i = np.square(M[:, i]-y_pred)
            score_i = (score_i-score_i.min())/(score_i.max()-score_i.min())
            score += score_i
        score = (score-score.min())/(score.max()-score.min())
        drop_index.append(np.argmax(score))
    return score
