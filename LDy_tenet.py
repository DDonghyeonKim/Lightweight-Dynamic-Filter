import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def TENet_arg_scope(is_training, weight_decay=0.00004, keep_prob=0.8):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.99,
        "activation_fn": None,
    }

    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout],
                                keep_prob=keep_prob,
                                is_training=is_training) as scope:
                return scope


def tenet(inputs, labels, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope):
    L = inputs.shape[1]
    C = inputs.shape[2]

    assert len(n_channels) == len(n_strides) + 1

    with tf.compat.v1.variable_scope(scope):

        def MM_augmentation(DY):
            DY = tf.reshape(DY, [-1, 98, 40, 1])
            T = tf.shape(DY)
            inds = tf.range(0, T[0])
            inds = tf.random_shuffle(inds)
            ORG = tf.gather(DY, inds, axis=0)
            N_fre = 40
            N_Time = 98

            Max_skips = 5
            N_skpis_fre = tf.range(Max_skips) + 2

            N_skpis_fre = tf.random_shuffle(N_skpis_fre)
            N_skpis_time = tf.range(Max_skips + 3) + 2
            N_skpis_time = tf.random_shuffle(N_skpis_time)

            FRe_bin = tf.range(N_fre - N_skpis_fre[0])
            FRe_bin = tf.random_shuffle(FRe_bin)
            Tim_bin = tf.range(N_Time - N_skpis_time[0])
            Tim_bin = tf.random_shuffle(Tim_bin)

            Fre_mask1 = tf.concat([tf.ones([1, FRe_bin[0], N_Time, 1]), tf.zeros([1, N_skpis_fre[0], N_Time, 1]),
                                   tf.ones([1, N_fre - FRe_bin[0] - N_skpis_fre[0], N_Time, 1])], 1)
            Tim_mask1 = tf.concat([tf.ones([1, N_fre, Tim_bin[0], 1]), tf.zeros([1, N_fre, N_skpis_time[0], 1]),
                                   tf.ones([1, N_fre, N_Time - Tim_bin[0] - N_skpis_time[0], 1])], 2)

            DY_mask = tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3])
            ORG_mask = (1 - tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3]))
            outputs = DY_mask * DY + (ORG_mask * (ORG + DY)) * 0.5

            return outputs

        def DSpec_arg_cut_mix(DY):
            DY = tf.reshape(DY, [-1, 98, 40, 1])
            T = tf.shape(DY)
            inds = tf.range(0, T[0])
            inds = tf.random_shuffle(inds)
            ORG = tf.gather(DY, inds, axis=0)
            N_fre = 40
            N_Time = 98

            Max_skips = 5
            N_skpis_fre = tf.range(Max_skips) + 2

            N_skpis_fre = tf.random_shuffle(N_skpis_fre)
            N_skpis_time = tf.range(Max_skips + 3) + 2
            N_skpis_time = tf.random_shuffle(N_skpis_time)

            FRe_bin = tf.range(N_fre - N_skpis_fre[0])
            FRe_bin = tf.random_shuffle(FRe_bin)
            Tim_bin = tf.range(N_Time - N_skpis_time[0])
            Tim_bin = tf.random_shuffle(Tim_bin)

            Fre_mask1 = tf.concat([tf.ones([1, FRe_bin[0], N_Time, 1]), tf.zeros([1, N_skpis_fre[0], N_Time, 1]),
                                   tf.ones([1, N_fre - FRe_bin[0] - N_skpis_fre[0], N_Time, 1])], 1)
            Tim_mask1 = tf.concat([tf.ones([1, N_fre, Tim_bin[0], 1]), tf.zeros([1, N_fre, N_skpis_time[0], 1]),
                                   tf.ones([1, N_fre, N_Time - Tim_bin[0] - N_skpis_time[0], 1])], 2)

            DY_mask = tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3])
            ORG_mask = -(1 - tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3]))
            outputs = DY_mask * DY + ORG_mask * ORG

            return outputs

        def Spec_arg(ORG):
            ORG = tf.reshape(ORG, [-1, 98, 40, 1])
            N_fre = 40
            N_Time = 98

            Max_skips = 5
            N_skpis_fre = tf.range(Max_skips) + 2

            N_skpis_fre = tf.random_shuffle(N_skpis_fre)
            N_skpis_time = tf.range(Max_skips + 3) + 2
            N_skpis_time = tf.random_shuffle(N_skpis_time)

            FRe_bin = tf.range(N_fre - N_skpis_fre[0])
            FRe_bin = tf.random_shuffle(FRe_bin)
            Tim_bin = tf.range(N_Time - N_skpis_time[0])
            Tim_bin = tf.random_shuffle(Tim_bin)

            Fre_mask1 = tf.concat([tf.ones([1, FRe_bin[0], N_Time, 1]), tf.zeros([1, N_skpis_fre[0], N_Time, 1]),
                                   tf.ones([1, N_fre - FRe_bin[0] - N_skpis_fre[0], N_Time, 1])], 1)
            Tim_mask1 = tf.concat([tf.ones([1, N_fre, Tim_bin[0], 1]), tf.zeros([1, N_fre, N_skpis_time[0], 1]),
                                   tf.ones([1, N_fre, N_Time - Tim_bin[0] - N_skpis_time[0], 1])], 2)

            DY_mask = tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3])
            # ORG_mask = -(1 - tf.transpose(Fre_mask1, [0, 2, 1, 3]) * tf.transpose(Tim_mask1, [0, 2, 1, 3]))
            outputs = DY_mask * ORG

            return outputs

        k = 9

        def DF(inputsss, weight, scopes):
            with tf.compat.v1.variable_scope(scopes):
                t = tf.shape(inputsss)
                # input=SPF(input)
                T_mean = tf.reduce_mean(inputsss, [1])

                # Patches_sf = tf.range(40)
                # Patches_sf = tf.random_shuffle(Patches_sf)

                # T_mean = tf.gather(params=T_mean, indices=Patches_sf, axis=1)

                Tweights0 = tf.matmul(T_mean, weight["W3"], name='mat1') + weight["B3"]

                # central_losss = center_loss(t[0], Tweights0, labels, centroids, centroids_delta)
                mean, vars = tf.nn.moments(Tweights0, [1], keep_dims=True)
                Tweights0 = tf.nn.batch_normalization(Tweights0, mean, vars, weight["LNa1"], weight["LNb1"], 0.001)
                Tweights2 = tf.matmul(tf.nn.relu(Tweights0), weight["W2"], name='mat3') + weight["B2"]  # batch,k
                # tt = tf.nn.relu(Tweights0)

                scale = tf.matmul(Tweights0, weight["W4"], name='mat4')
                bia = tf.matmul(Tweights0, weight["W5"], name='mat5')
#                Z1=(s1-mean_1)

#                Z2=(s2-mean_2)

#                Z1_cov=tf.matmul(tf.transpose(Z1,[1,0]),Z1)/99
                #Z2_cov=tf.matmul(tf.transpose(Z2,[1,0]),Z2)/99

#                Zero_filter=tf.nn.relu(tf.ones([256,256])-tf.eye(256))
#                cov_loss =tf.reduce_sum(tf.square(Zero_filter*Z1_cov)) + tf.reduce_sum(tf.square(Zero_filter*Z2_cov))

                #

                # Covs = tf.matmul(cov_w, cov_w, transpose_a=True)

                # Cos_loss=tf.reduce_mean(tf.log((tf.nn.relu(1+(cos_similarity)*(1-LB))+0.00001)/(tf.nn.relu(cos_similarity*LB))+0.00001))

                weights = tf.reshape(Tweights2, [-1, 1, 1, 9])

                patches = tf.extract_image_patches(tf.expand_dims(inputsss, 3), ksizes=[1, 3, 3, 1],
                                                   strides=[1, 1, 1, 1],
                                                   rates=[1, 2, 2, 1],
                                                   padding='SAME')
                # Pixels_sf = tf.range(9)
                # Pixels_sf = tf.random_shuffle(Pixels_sf)
                # patches2 = tf.gather(params=patches, indices=Pixels_sf, axis=3)

                # TF_weights = tf.reduce_sum(patches*tf.reshape(weight["W1"],[1,1,1,9]),axis=3)

                def gaussian_noise_layer(input_layer, std=0.2):
                    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
                    inds = tf.random_normal(shape=[1], mean=0.0, stddev=0.5, dtype=tf.float32)
                    return tf.nn.relu(inds)*noise

                TF_weights = tf.nn.conv2d(tf.expand_dims(inputsss, 3), weight["W1"], strides=[1, 1, 1, 1],
                                          dilations=[1, 2, 2, 1], padding="SAME")
                mean, vars = tf.nn.moments(TF_weights, [1], keep_dims=True)
                TF_weights = tf.nn.batch_normalization(TF_weights, mean, vars, weight["LNa3"], weight["LNb3"], 0.001)
                # TF_weights2 = tf.nn.conv2d(tf.nn.relu(TF_weights), weights2["W0"], strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], padding="SAME")

                D_weights = weights * tf.sigmoid(TF_weights)
                #D_weights = D_weights + gaussian_noise_layer(D_weights)

#                s_weight=tf.sigmoid(TF_weights)

#                s_weight=tf.reshape(s_weight,[-1,98*40])

#                means_dev, vars_dev = tf.nn.moments(tf.sigmoid(TF_weights), [1], keep_dims=True)

#                v_loss=tf.reduce_mean(tf.nn.relu(1-tf.sqrt(vars_dev+0.0001)))

                Feats = tf.reduce_sum(patches * D_weights, 3)

                mean, vars = tf.nn.moments(Feats, [1], keep_dims=True)

                S1=((Feats-mean)/(tf.sqrt(vars)+0.00001))*tf.reshape(scale,[-1,1,40]) + tf.reshape(bia,[-1,1,40])
                ##Activation

                Feats2=S1+inputsss
                meanss=tf.reduce_mean(Feats2,1)


            return Feats2,tf.reduce_sum(tf.abs(D_weights))

        def DF1(Input_Feat, W, W2):

            t = tf.shape(Input_Feat)
            inpatches = tf.extract_image_patches(tf.expand_dims(Input_Feat, 3), ksizes=[1, 10, 1, 1],
                                                 strides=[1, 10, 1, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')  ##-1,160,1,100
            inpatches = tf.transpose(inpatches, [0, 2, 3, 1])  # -1,F,T,C (-1,40,10,10)
            inpatches_P = tf.extract_image_patches(inpatches, ksizes=[1, 2, 2, 1],
                                                   strides=[1, 1, 2, 1],
                                                   rates=[1, 2, 2, 1],
                                                   padding='SAME')  ##-1,160,1,100
            Conv1 = (tf.reduce_sum(tf.reshape(inpatches_P, [-1, 40, 5, 4, 10]) * W["W1"], 3))  ##-1,40,5,10
            # Conv1=tf.nn.relu(tf.reduce_sum(tf.reshape(inpatches_P,[-1,40,5,9,10])*W["W1"],3))##-1,40,5,10
            mean, vars = tf.nn.moments(Conv1, [3], keep_dims=True)
            Conv1 = tf.nn.batch_normalization(Conv1, mean, vars, W["LNa1"], W["LNb1"], 0.001)
            inpatches_P2 = tf.extract_image_patches(tf.transpose(Conv1, [0, 1, 3, 2]), ksizes=[1, 2, 2, 1],
                                                    strides=[1, 1, 2, 1],
                                                    rates=[1, 2, 2, 1],
                                                    padding='SAME')  ##-1,160,1,100

            Conv2 = (tf.reduce_sum(tf.reshape(inpatches_P2, [-1, 40, 5, 4, 5]) * W["W2"], 3))  ##-1,40,5,10
            # Conv2=tf.nn.relu(tf.reduce_sum(tf.reshape(inpatches_P2,[-1,40,5,9,5])*W["W2"],3))##-1,40,5,10
            mean, vars = tf.nn.moments(Conv2, [3], keep_dims=True)
            Conv22 = tf.nn.batch_normalization(Conv2, mean, vars, W["LNa2"], W["LNb2"], 0.001)
            LAP = tf.reshape(Conv22, [-1, 40, 25, 1])
            Gate = tf.sigmoid(tf.nn.conv2d(LAP, W["W4"], strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 2, 2, 1]))
            CVs = tf.nn.conv2d(LAP, W["W6"], strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 2, 2, 1])
            AAP = tf.reduce_sum(tf.tanh(Gate) * CVs, [2, 3])

            # Convs = tf.reduce_sum(LAP*W["W4"],2,keep_dims=True)#-1,1,T
            # GAP=tf.reduce_mean(tf.reshape(Conv22,[-1,40,25]),2,keep_dims=True)
            # Atts = tf.nn.softmax(tf.reduce_sum(Convs * LAP, 1, keep_dims=True), axis=2)
            # AAP = tf.reduce_sum(LAP * Atts, 2)
            # GAP=tf.concat([mean, tf.square(vars+0.00001)],1)

            K_filter = tf.reshape(tf.sigmoid(tf.matmul(AAP, W["W3"])), [-1, 1, 1, 9])
            ###gaussian_pooling

            Inp_patch2 = tf.extract_image_patches(tf.expand_dims(Input_Feat[:, 1:99, :], 3), ksizes=[1, 3, 3, 1],
                                                  strides=[1, 1, 1, 1],
                                                  rates=[1, 2, 2, 1],
                                                  padding='SAME')  # -1,14,10,28
            Tconv3 = tf.reduce_sum(Inp_patch2 * W2["W1"], 3)
            Tconv3 = tf.reshape(Tconv3, [-1, 49, 2, 40])
            mean, vars = tf.nn.moments(Tconv3, [1], keep_dims=True)
            Tconv3 = tf.nn.batch_normalization(Tconv3, mean, vars, W2["LNa1"], W2["LNb1"], 0.001)  # -1,40,100
            Tconv3 = tf.reshape(Tconv3, [-1, 98, 40, 1])

            D_weights = Tconv3 * tf.reshape((K_filter), [-1, 1, 1, 9])
            Feats = (tf.reduce_sum(Inp_patch2 * D_weights, 3))

            Feats = tf.reshape(Feats, [-1, 49, 2, 40])
            mean, vars = tf.nn.moments(Feats, [1], keep_dims=True)
            Feats2 = tf.reshape(tf.nn.batch_normalization(Feats, mean, vars, W2["LNa2"], W2["LNb2"], 0.001),
                                [-1, 98, 40]) + Input_Feat[:, 1:99, :]
            return Feats2

        weights1 = {
            # "W10": tf.Variable(tf.truncated_normal([7, 7,9, 40], stddev=0.01), name="W0"),
#            "W0": tf.Variable(tf.truncated_normal([9, 40], stddev=0.01), name="W00"),
            "W1": tf.Variable(tf.truncated_normal([3, 3, 1, 1], stddev=0.01), name="W1"),
            "W2": tf.Variable(tf.truncated_normal([40, k], stddev=0.01), name="W2"),
            "W3": tf.Variable(tf.truncated_normal([40, 40], stddev=0.01), name="W3"),
            "W4": tf.Variable(tf.truncated_normal([40, 40], stddev=0.01), name="W4"),
            "W5": tf.Variable(tf.truncated_normal([40, 40], stddev=0.01), name="W5"),
            "B2": tf.Variable(tf.truncated_normal([1, k], stddev=0.01), name="B2"),
            "B3": tf.Variable(tf.truncated_normal([1, 40], stddev=0.01), name="B3"),
            "LNa2": tf.Variable(tf.truncated_normal([1, 1, 40], stddev=0.01), name="LNa2"),
            "LNb2": tf.Variable(tf.truncated_normal([1, 1, 40], stddev=0.01), name="LNb2"),
            "LNa3": tf.Variable(tf.truncated_normal([1, 1, 40, 1], stddev=0.01), name="LNa3"),
            "LNb3": tf.Variable(tf.truncated_normal([1, 1, 40, 1], stddev=0.01), name="LNb3"),
            "LNa1": tf.Variable(tf.truncated_normal([1, 40], stddev=0.01), name="LNa1"),
            "LNb1": tf.Variable(tf.truncated_normal([1, 40], stddev=0.01), name="LNb1"),
            "DE": tf.Variable(tf.truncated_normal([40, 40 * 10], stddev=0.01), name="DE"),
            "LNb0": tf.Variable(tf.truncated_normal([1, 1, 40], stddev=0.01), name="LNb0"),
        }



#        DFB1=DF(inputs_re, weights2)

#        inputs=Spec_arg(inputs)
#        inputs_new=MM_augmentation(inputs)
        inputs_re = tf.reshape(inputs, [-1, 98, 40])
        #inputs_new = tf.reshape(inputs_new, [-1, 98, 40])
        t = tf.shape(inputs_re)

        # inputs_rer=tf.squeeze(MM_augmentation(inputs_re),3)

        # t=tf.shape(inputs_re)

        #        out1=DF1(tf.concat([Pad,inputs_re,Pad],1),Trans,Trans2)
        # out1=Spec_arg(tf.expand_dims(inputs_re,3))
        # out1 =tf.squeeze(out1,3)
        def gaussian_noise_layer(input_layer, std):
            noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
            return input_layer + noise


        out1,KL_divergence = DF(inputs_re, weights1, "D1")
#        out_new = DF(gaussian_noise_layer(inputs_re,0.1), weights1, "D1")


        #out2=tf.nn.leaky_relu(tf.nn.conv2d(tf.expand_dims(out1,3),FL["W1"],strides=[1, 2,2, 1],padding="SAME"))
        #out3=tf.nn.leaky_relu(tf.nn.conv2d(out2,FL["W2"],strides=[1, 2,2, 1],padding="SAME"))
        #out4=(tf.nn.conv2d(out3,FL["W3"],strides=[1, 1,1, 1],padding="SAME"))

#        out2_new=tf.nn.leaky_relu(tf.nn.conv2d(tf.expand_dims(out_new,3),FL["W1"],strides=[1, 2,2, 1],padding="SAME"))
#        out3_new=tf.nn.leaky_relu(tf.nn.conv2d(out2_new,FL["W2"],strides=[1, 2,2, 1],padding="SAME"))
#        out4_new=(tf.nn.conv2d(out3_new,FL["W3"],strides=[1, 1,1, 1],padding="SAME"))

#        Ts1=tf.reshape(out4,[-1,25*10,256])
#        Ts2=tf.reshape(out4_new,[-1,25*10,256])

#        S = tf.shape(Ts1)
#        inds = tf.range(0, S[1])
#        inds = tf.random_shuffle(inds)
#        ORG = tf.gather(Ts1, inds, axis=1)
#        Fake = tf.gather(Ts2, inds, axis=1)

#        Postive=ORG[:,:10,:]
#        Negtive=tf.expand_dims(Fake[:,0,:],1)

        #Postive=Postive/(tf.sqrt(tf.reduce_sum(tf.square(Postive),2,keep_dims=True))+0.0001)
        #Negtive=Negtive/(tf.sqrt(tf.reduce_sum(tf.square(Negtive),2,keep_dims=True))+0.0001)
        #Pairs=tf.concat([Postive,Negtive],1)
        #C=tf.reshape(tf.matmul(Postive,tf.transpose(Negtive,[0,2,1])),[-1,10])/0.07



        #tpr_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        #    labels=tf.zeros([S[0]],dtype='int32'), logits=C,
        #)
        #Z=tf.expand_dims(0.001+out2-tf.reduce_mean(out2,0,keep_dims=True),2)
        #C=tf.reduce_mean(tf.matmul(tf.transpose(Z,[0,2,1]),Z),0)
        #tpr_loss=(tf.reduce_sum(tf.square(C))-tf.reduce_sum(tf.square(tf.eye(256)*C)))*0.5
        #tpr_loss=tf.reduce_mean(tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=labels,embeddings=tf.reduce_mean(out4,[1,2])))



        #VAR_loss=(tf.reduce_mean(tf.square(recons1-inputs_re))+tf.reduce_mean(tf.square(recons_new-out_new))) * 0.1
#        out2 = DF(tf.reverse(inputs_re,[1]), weights1, "D1")


#        var_loss=tf.reduce_mean(tf.nn.relu(1-tf.sqrt(vars_1+0.0001))) + tf.reduce_mean(tf.nn.relu(1-tf.sqrt(vars_2+0.0001)))




#        Z1=(s1-mean_1)

#        Z2=(s2-mean_2)

#        Z1_cov=tf.matmul(tf.transpose(Z1,[1,0]),Z1)/99
#        Z2_cov=tf.matmul(tf.transpose(Z2,[1,0]),Z2)/99

#        Zero_filter=tf.nn.relu(tf.ones([256,256])-tf.eye(256))
#        cov_loss =tf.reduce_sum(tf.square(Zero_filter*Z1_cov)) + tf.reduce_sum(tf.square(Zero_filter*Z2_cov))

#        VAR_loss= MSE_loss*0.1+var_loss*0.1


        #mean, vars = tf.nn.moments(out1, [1,2], keep_dims=True)



#        out4=(out1+inputs_re)*tf.nn.softmax(out2,2)*tf.nn.softmax(out3,1)




#        out3=(out1*tf.sigmoid(out2))+inputs_re

#        out3=out1*tf.sigmoid(out2)
#        out3 = (DF(out2, weights3, "D3"))
        #out6 = tf.nn.dropout(out6, 0.9)
        inputs2 = tf.expand_dims(out1, 2)  # [N, L, 1, C]

        first_conv_kernel = [3, 1]
        conv_kernel = [9, 1]

        net = slim.conv2d(
            inputs2, num_outputs=n_channels[0], kernel_size=first_conv_kernel, stride=1, scope="conv0")

        n_channels = n_channels[1:]

        for i, n in enumerate(n_channels):
            with tf.compat.v1.variable_scope(f"block{i}"):
                expand_n = int(n * n_ratios[i])
                for j, channel in enumerate(range(n_layers[i])):
                    stride = n_strides[i] if j == 0 else 1
                    if stride != 1 or net.shape[-1] != n:
                        layer_in = slim.conv2d(
                            net, num_outputs=n, activation_fn=None, kernel_size=1, stride=stride, scope=f"down")
                    else:
                        layer_in = net

                    net = slim.conv2d(net,
                                      expand_n,
                                      kernel_size=[1, 1],
                                      scope=f"pointwise_conv{j}_0")
                    if kernel_list:
                        list_net = []
                        for k, kernel_size in enumerate(kernel_list):
                            list_net.append(slim.separable_convolution2d(net,
                                                                         num_outputs=None,
                                                                         activation_fn=None,
                                                                         stride=stride,
                                                                         depth_multiplier=1,
                                                                         kernel_size=[
                                                                             kernel_size, 1],
                                                                         scope=f"depthwise_conv{j}_{k}"))
                        net = tf.add_n(list_net)
                        net = tf.nn.relu(net)
                    else:
                        net = slim.separable_convolution2d(net,
                                                           num_outputs=None,
                                                           stride=stride,
                                                           depth_multiplier=1,
                                                           kernel_size=conv_kernel,
                                                           scope=f"depthwise_conv{j}")
                    net = slim.conv2d(net, n, activation_fn=None, kernel_size=[
                        1, 1], scope=f"pointwise_conv{j}_1")
                    net += layer_in

        # Supervisions = net[t[0]:, :, :, :]
        KWs = net

        net2 = slim.avg_pool2d(
            KWs, kernel_size=KWs.shape[1:3], stride=1, scope="avg_pool")

        net2 = slim.dropout(net2)

        logits = slim.conv2d(
            net2, num_classes, 1, activation_fn=None, normalizer_fn=None, scope="fc")
        logits = tf.reshape(
            logits, shape=(-1, logits.shape[3]), name="squeeze_logit")

    return logits,KL_divergence


def TENet12(inputs, labels, num_classes, kernel_list, scope):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet12"
        else:
            scope = "TENet12"

    return tenet(inputs, labels, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet6(inputs, labels, num_classes, kernel_list, scope):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet6"
        else:
            scope = "TENet6"

    return tenet(inputs, labels, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet12Narrow(inputs, labels, num_classes, kernel_list, scope):
    n_channels = [16] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet12Narrow"
        else:
            scope = "TENet12Narrow"

    return tenet(inputs, labels, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet6Narrow(inputs, labels, num_classes, kernel_list, scope):
    n_channels = [16] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet6Narrow"
        else:
            scope = "TENet6Narrow"

    return tenet(inputs, labels, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)
