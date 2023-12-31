from layers import *
from tensorflow.keras import Model, metrics, Sequential


def get_conv_AE(input_shape, patch_size, embed_dim, num_stages):
    I = layers.Input(input_shape, name="INPUT")
    dim = int(input_shape[1] / patch_size[1])
    e1 = CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding="SAME",
        name=f"first_cir",
    )(I)
    e2 = CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding="SAME",
        name=f"2nd_cir",
    )(e1)
    #     e = layers.Conv3D(filters=embed_dim,kernel_size=patch_size[0],strides=patch_size[0], use_bias=True, padding="SAME", name=f"e_first")(I)
    e = e2
    encoder_stages = []
    for i in range(num_stages):
        e = Double_CIR(
            filters=embed_dim * 2**i,
            kernel_size=min(dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"e{i}_cir",
        )(e)
        encoder_stages.append(e)
        e = layers.MaxPool3D(pool_size=2, name=f"e{i}_mp")(e)
        dim = max(dim // 2, 1)

    e = Double_CIR(
        filters=embed_dim * 2 ** (num_stages),
        kernel_size=min(dim, 3),
        strides=1,
        use_bias=False,
        padding="SAME",
        name=f"e{num_stages}_cir",
    )(e)

    for i in range(1, num_stages + 1):
        dim *= 2
        e = layers.Conv3DTranspose(
            filters=embed_dim * 2 ** (num_stages - i),
            kernel_size=2,
            strides=2,
            name=f"d{i}_up",
        )(e)
        e = Double_CIR(
            filters=embed_dim * 2 ** (num_stages - i),
            kernel_size=min(dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"d_{i}_cir",
        )(e)

    e = layers.Concatenate(name="conc1")([e, e2])
    e = Double_CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding="SAME",
        name=f"d{num_stages+1}_cir",
    )(e)
    e = layers.Conv3DTranspose(
        filters=embed_dim,
        kernel_size=patch_size[1] // 2,
        strides=patch_size[1] // 2,
        name=f"d{num_stages+1}_up",
    )(e)

    e = layers.Concatenate(name="conc2")([e, e1])
    e = Double_CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding="SAME",
        name=f"d{num_stages+2}_cir",
    )(e)
    e = layers.Conv3DTranspose(
        filters=embed_dim,
        kernel_size=patch_size[1] // 2,
        strides=patch_size[1] // 2,
        name=f"d{num_stages+2}_up",
    )(e)

    e = layers.Conv3D(
        filters=1, kernel_size=3, use_bias=False, padding="SAME", name=f"conv_last"
    )(e)

    model = Model(inputs=[I], outputs=[*encoder_stages, e])
    return model


def get_conv_encoder(input_shape, patch_size, embed_dim, num_stages):
    I = layers.Input(input_shape, name="INPUT")
    dim = int(input_shape[1] / patch_size[1])
    e1 = CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding="SAME",
        name=f"first_cir",
    )(I)
    e2 = CIR(
        filters=embed_dim,
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding="SAME",
        name=f"2nd_cir",
    )(e1)
    #     e = layers.Conv3D(filters=embed_dim,kernel_size=patch_size[0],strides=patch_size[0], use_bias=True, padding="SAME", name=f"e_first")(I)
    e = e2
    encoder_stages = [e1]
    for i in range(num_stages):
        e = Double_CIR(
            filters=embed_dim * 2**i,
            kernel_size=min(dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"e{i}_cir",
        )(e)
        encoder_stages.append(e)
        e = layers.MaxPool3D(pool_size=2, name=f"e{i}_mp")(e)
        dim = max(dim // 2, 1)

    e = Double_CIR(
        filters=embed_dim * 2 ** (num_stages),
        kernel_size=min(dim, 3),
        strides=1,
        use_bias=False,
        padding="SAME",
        name=f"e{num_stages}_cir",
    )(e)
    model = Model(inputs=[I], outputs=[*encoder_stages, e])
    return model


def get_swin_AE(
    input_shape,
    embed_dim,
    window_size,
    patch_size,
    mask_ratio,
    depths,
    mlp_ratio,
    num_heads,
    patch_norm=False,
    qkv_bias=True,
    drop_path=0.0,
    attn_drop=0.0,
    proj_drop=0.0,
):
    dpr = np.linspace(0, drop_path, sum(depths))
    num_stages = len(depths)
    bottleneck_dim = max(
        int(input_shape[1] / (patch_size[0] * 2**num_stages)), 1)
    I = layers.Input(input_shape, name="input")
    #     x, mask_indeces = Masked_PatchEmbedding(patch_size=patch_size, mask_ratio=mask_ratio, embed_dim=embed_dim, use_norm=True,name="Patch_Embed")(I)
    x = PatchEmbedding(
        patch_size=patch_size, embed_dim=embed_dim, use_norm=True, name="Patch_Embed"
    )(I)
    # x = layers.Dropout(rate=proj_drop,name="drop1")(x) # We've already masked some tokens
    e = x
    swin_enc_outs = []
    for i in range(num_stages):
        e = SwinTransformerStage(
            depth=depths[i],
            embed_dim=embed_dim * (2**i),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads[i],
            qkv_bias=qkv_bias,
            drop_path=dpr[i],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            name=f"s{i}",
        )(e)
        o = layers.LayerNormalization(name=f"out_norm_{i}")(e)
        swin_enc_outs.append(o)
    #         break
    p = Double_CIR(
        filters=embed_dim * 16,
        kernel_size=min(bottleneck_dim, 3),
        strides=1,
        use_bias=False,
        padding="SAME",
        name="p3",
    )(swin_enc_outs[-1])
    p = layers.Conv3DTranspose(
        filters=embed_dim * 8, kernel_size=2, strides=2, name="e3_up"
    )(p)
    for i in range(1, 4):
        d = layers.Concatenate(
            axis=-1, name=f"conc{3-i}")([p, swin_enc_outs[3 - i]])
        p = Double_CIR(
            filters=embed_dim * 2 ** (4 - i),
            kernel_size=min((2**i) * bottleneck_dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"p{3-i}",
        )(d)
        p = layers.Conv3DTranspose(
            filters=embed_dim * 2 ** (3 - i),
            kernel_size=2,
            strides=2,
            name=f"p{3-i}_up",
        )(p)

    d = layers.Concatenate(axis=-1, name="conc10")([p, x])
    p = Double_CIR(
        filters=embed_dim * 1,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding="SAME",
        name="p10",
    )(d)
    p_up = layers.Conv3DTranspose(
        filters=embed_dim * 1, kernel_size=patch_size, strides=patch_size, name="p10_up"
    )(p)
    c_1 = Double_CIR(
        filters=embed_dim * 1,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding="SAME",
        name="p",
    )(p_up)
    out = layers.Conv3D(filters=1, kernel_size=3,
                        padding="same", name="last_p")(c_1)
    return Model(inputs=I, outputs=[out])


def get_XNet(
    input_shape,
    embed_dim,
    window_size,
    patch_size,
    mask_ratio,
    depths,
    mlp_ratio,
    num_heads,
    patch_norm=True,
    qkv_bias=True,
    drop_path=0.0,
    attn_drop=0.0,
    proj_drop=0.0,
):
    dpr = np.linspace(0, drop_path, sum(depths))
    num_stages = len(depths)
    bottleneck_dim = max(
        int(input_shape[1] / (patch_size[0] * 2**num_stages)), 1)
    I = layers.Input(input_shape, name="input")
    conv_AE = get_conv_AE(input_shape, patch_size, embed_dim, num_stages)
    out = conv_AE(I)
    conv_encoder_outs, conv_AE_out = out[:-1], out[-1]

    #     return Model(inputs=I,outputs=[conv_AE_out])

    #     x, mask_indeces = Masked_PatchEmbedding(patch_size=patch_size, mask_ratio=mask_ratio, embed_dim=embed_dim, use_norm=True,name="Patch_Embed")(I)
    x = PatchEmbedding(
        patch_size=patch_size,
        embed_dim=embed_dim,
        use_norm=patch_norm,
        name="Patch_Embed",
    )(I)
    # x = layers.Dropout(rate=proj_drop,name="drop1")(x) # We've already masked some tokens
    e = x
    for i in range(num_stages):
        e = SwinTransformerStage(
            depth=depths[i],
            embed_dim=embed_dim * (2**i),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads[i],
            qkv_bias=qkv_bias,
            drop_path=dpr[i],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            name=f"s{i}",
        )(e)
    e = Double_CIR(
        filters=embed_dim * 16,
        kernel_size=min(bottleneck_dim, 3),
        strides=1,
        use_bias=False,
        padding="SAME",
        name="e",
    )(e)
    d = layers.Conv3DTranspose(
        filters=embed_dim * 8, kernel_size=2, strides=2, name="e3_up"
    )(e)

    for i in range(4):
        d = layers.Concatenate(axis=-1, name=f"conc{3-i}")(
            [d, conv_encoder_outs[3 - i]]
        )
        d = Double_CIR(
            filters=embed_dim * 2 ** (3 - i),
            kernel_size=min((2**i) * bottleneck_dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"d{3-i}",
        )(d)
        d = layers.Conv3DTranspose(
            filters=embed_dim * 2 ** (3 - i),
            kernel_size=2,
            strides=2,
            name=f"p{3-i}_up",
        )(d)

    d = layers.Conv3DTranspose(
        filters=embed_dim * 2 ** (3 - i), kernel_size=2, strides=2, name="p_up"
    )(d)
    swin_AE_out = layers.Conv3D(
        filters=1, kernel_size=3, padding="same", name="o")(d)
    return Model(inputs=I, outputs=[conv_AE_out, swin_AE_out])


def get_XNet_V2(
    input_shape,
    embed_dim,
    window_size,
    patch_size,
    mask_ratio,
    depths,
    mlp_ratio,
    num_heads,
    patch_norm=True,
    qkv_bias=True,
    drop_path=0.0,
    attn_drop=0.0,
    proj_drop=0.0,
):
    dpr = np.linspace(0, drop_path, sum(depths))
    num_stages = len(depths)
    bottleneck_dim = max(
        int(input_shape[1] / (patch_size[0] * 2**num_stages)), 1)
    I = layers.Input(input_shape, name="input")
    conv_AE = get_conv_encoder(input_shape, patch_size, embed_dim, num_stages)
    out = conv_AE(I)
    conv_encoder_outs, conv_AE_out = out[:-1], out[-1]
    x = PatchEmbedding(
        patch_size=patch_size,
        embed_dim=embed_dim,
        use_norm=patch_norm,
        name="Patch_Embed",
    )(I)
    e = x
    for i in range(num_stages):
        e = SwinTransformerStage(
            depth=depths[i],
            embed_dim=embed_dim * (2**i),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads[i],
            qkv_bias=qkv_bias,
            drop_path=dpr[i],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            name=f"s{i}",
        )(e)
    e = Double_CIR(
        filters=embed_dim * 16,
        kernel_size=min(bottleneck_dim, 3),
        strides=1,
        use_bias=False,
        padding="SAME",
        name="e",
    )(e)
    # bottleneck_features = layers.Concatenate(
    #     name="bottleneck_conc")([conv_AE_out, e])
    cls = layers.Dense(name="bottleneck_fc1", units=256, activation="leaky_relu")(
        conv_AE_out
    )
    cls = layers.Dense(name="bottleneck_fc2", units=1)(cls)
    cls = layers.Reshape(name="reshape", target_shape=(1,))(cls)
    d = layers.Conv3DTranspose(
        filters=embed_dim * 8, kernel_size=2, strides=2, name="e3_up"
    )(e)

    for i in range(5):
        d = layers.Concatenate(axis=-1, name=f"conc{4-i}")(
            [d, conv_encoder_outs[4 - i]]
        )
        d = Double_CIR(
            filters=embed_dim * 2 ** (4 - i),
            kernel_size=min((2**i) * bottleneck_dim, 3),
            strides=1,
            use_bias=False,
            padding="SAME",
            name=f"d{4-i}",
        )(d)
        d = layers.Conv3DTranspose(
            filters=embed_dim * 2 ** (4 - i),
            kernel_size=2,
            strides=2,
            name=f"p{4-i}_up",
        )(d)

    swin_AE_out = layers.Conv3D(
        filters=1, kernel_size=3, padding="same", name="o")(d)
    return Model(inputs=I, outputs=[swin_AE_out, cls])


class XNet(object):
    def __init__(self, optimizer, conv_loss, swin_loss, **model_args):
        self.model = get_XNet(**model_args)
        self.optim = optimizer
        self.conv_loss = conv_loss
        self.swin_loss = swin_loss

    @tf.function
    def train_step(self, x, *args):
        with tf.GradientTape() as tape:
            conv_out, swin_out = self.model(x)
            l1 = self.swin_loss(args[0], swin_out)
            l2 = self.conv_loss(args[0], conv_out)
            tr_loss = l1 + 3.0 * l2
        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(tr_loss, trainable_vars)
        self.optim.apply_gradients(zip(grads, trainable_vars))
        return tr_loss

    @tf.function
    def val_step(self, x, *args):
        conv_out, swin_out = self.model(x)
        l1 = self.swin_loss(args[0], swin_out)
        l2 = self.conv_loss(args[0], conv_out)
        val_loss = l1 + 3.0 * l2
        return val_loss


class XNet_v2(object):
    def __init__(self, optimizer, cls_loss, swin_loss, **model_args):
        self.model = get_XNet_V2(**model_args)
        self.optim = optimizer
        self.cls_loss = cls_loss
        self.swin_loss = swin_loss
        self.metrics = {"tr_loss": metrics.Mean(name="tr_loss"),
                        "vl_loss": metrics.Mean(name="vl_loss"),
                        "tr_iou": metrics.BinaryIoU(target_class_ids=[1], name="tr_iou"),
                        "vl_iou": metrics.BinaryIoU(target_class_ids=[1], name="vl_iou"),
                        "tr_rec": metrics.Recall(name="tr_rec"),
                        "tr_pre": metrics.Precision(name="tr_pre"),
                        "vl_rec": metrics.Recall(name="vl_rec"),
                        "vl_pre": metrics.Precision(name="vl_pre")
                        }

    @tf.function
    def train_step(self, x, *args):
        with tf.GradientTape() as tape:
            swin_out, cls = self.model(x)
            l1 = self.swin_loss(args[0], swin_out)
            l2 = self.cls_loss(args[1], cls)
            tr_loss = 0.7 * l1 + 0.3 * l2
        trainable_vars = self.model.trainable_variables
        grads = tape.gradient(tr_loss, trainable_vars)
        self.optim.apply_gradients(zip(grads, trainable_vars))
        self.metrics["tr_loss"].update_state(tr_loss)
        self.metrics["tr_iou"].update_state(args[0], tf.nn.sigmoid(swin_out))
        self.metrics["tr_rec"].update_state(args[1], tf.nn.sigmoid(cls))
        self.metrics["tr_pre"].update_state(args[1], tf.nn.sigmoid(cls))

    @tf.function
    def val_step(self, x, *args):
        swin_out, cls = self.model(x)
        l1 = self.swin_loss(args[0], swin_out)
        l2 = self.cls_loss(args[1], cls)
        val_loss = 0.7 * l1 + 0.3 * l2
        self.metrics["vl_loss"].update_state(val_loss)
        self.metrics["vl_iou"].update_state(args[0], tf.nn.sigmoid(swin_out))
        self.metrics["vl_rec"].update_state(args[1], tf.nn.sigmoid(cls))
        self.metrics["vl_pre"].update_state(args[1], tf.nn.sigmoid(cls))


def patch_lstm_v3(input_shape, df=8):
    pad = 'SAME'
    act = 'relu'
    ks = 3
    st = 1
    I = layers.Input(input_shape, name='Input')
    a = T_CIR(df, ks, st, use_norm=1, time_dist=True,
              mode="conv", name="c0")(I)
    a = layers.MaxPooling3D(pool_size=(1, 2, 2), name="p1")(a)

    a = T_CIR(2*df, ks, st, use_norm=1, time_dist=True,
              mode="conv", name="c1")(a)
    a = layers.MaxPooling3D(pool_size=(1, 2, 2), name="p2")(a)

    a = T_CIR(4*df, ks, st, use_norm=1, time_dist=True,
              mode="conv", name="c2")(a)
    a = layers.MaxPooling3D(pool_size=(1, 2, 2), name="p3")(a)

    a = layers.ConvLSTM2D(4*df, kernel_size=1, strides=1, activation=act,
                          padding=pad, kernel_initializer="glorot_normal", return_sequences=False)(a)
    a = tfa.layers.InstanceNormalization()(a)
    a = layers.MaxPooling2D(pool_size=(2, 2), name="p4")(a)

    a = T_CIR(4*df, ks, st, use_norm=1, time_dist=False,
              mode="conv", name="c3")(a)
    a = T_CIR(2*df, ks, st, use_norm=1,
              time_dist=False, mode="conv", name="c4")(a)
    a = layers.GlobalAveragePooling2D(name="gap")(a)
    a = layers.Dense(units=df, activation="relu")(a)
    # a = tfa.layers.InstanceNormalization()(a)
    a = layers.Dense(units=1, activation="sigmoid", name="fc")(a)
    m = Model(I, a, name='patch_lstm_v3')
    return m


def get_ConvLSTM_Model(input_shape, base_filters, init, act, ks, st):
    I = layers.Input(input_shape)
    a = layers.ConvLSTM2D(base_filters, kernel_size=ks, strides=st, activation=act,
                          padding='SAME', kernel_initializer=init, return_sequences=True, name='ConvLSTM1')(I)
    # a = layers.ConvLSTM2D(2*base_filters, kernel_size=ks, strides=st, activation=act,
    #                       padding='SAME', kernel_initializer=init, return_sequences=True, name='ConvLSTM2')(a)
    a = layers.ConvLSTM2D(base_filters, kernel_size=ks, strides=st, activation=act,
                          padding='SAME', kernel_initializer=init, return_sequences=False, name='ConvLSTM3')(a)
    a = layers.GlobalAveragePooling2D(name='GAP')(a)
    a = layers.Dense(units=1, activation='linear',
                     use_bias=False, kernel_initializer=init, name='FC')(a)

    return Model(inputs=I, outputs=[a])


def get_DeformConv_Model(input_shape, ks, base_filters, num_blocks, init, act, conv_mode="deform_conv"):
    I = layers.Input(input_shape)
    # o = DCNv2(base_filters, 3, activation=act,
    #           use_bias=True, kernel_initializer=init)(I)

    L = [I]
    for i in range(num_blocks):
        # L.append(dcn_v1(base_filters, ks, num_deformable_group=1,
        #                 padding="SAME", activation=act, use_bias=True, kernel_initializer=init))

        if conv_mode == "deform_conv_v1":
            c = DCN_v1(base_filters, ks, padding="SAME",
                       use_bias=False, kernel_initializer=init)
        elif conv_mode == "deform_conv_v2":
            c = DCNv2(base_filters, ks, use_bias=False,
                      kernel_initializer=init)
        else:
            c = layers.Conv2D(
                base_filters, ks, use_bias=False, kernel_initializer=init, padding="SAME")
        L.append(c)
        L.append(layers.LayerNormalization())
        L.append(layers.LeakyReLU())
        L.append(layers.MaxPooling2D((2, 2)))
        base_filters *= 2

    return Sequential(L)


class abus_cls(object):
    def __init__(self, batch_size, input_shape, base_filters, act, ks, st, num_conv_blocks, conv_type, init, loss_fn, optim, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.deform_model = get_DeformConv_Model(
            input_shape=input_shape[1:], base_filters=base_filters, ks=ks, num_blocks=num_conv_blocks, init=init, act=act, conv_mode=conv_type)
        covnlstm_input_shape = (input_shape[0], input_shape[1]//2**num_conv_blocks,
                                input_shape[2]//2**num_conv_blocks, base_filters*2**(num_conv_blocks-1))
        self.convlstm_model = get_ConvLSTM_Model(
            input_shape=covnlstm_input_shape, base_filters=base_filters*2**num_conv_blocks, init=init, act=act, ks=ks, st=st)
        self.loss_fn = loss_fn
        self.optim = optim
        self.metrics = {"tr_loss": metrics.Mean(name="tr_loss"),
                        "vl_loss": metrics.Mean(name="vl_loss"),
                        "tr_rec": metrics.Recall(name="tr_rec"),
                        "tr_pre": metrics.Precision(name="tr_pre"),
                        "vl_rec": metrics.Recall(name="vl_rec"),
                        "vl_pre": metrics.Precision(name="vl_pre")
                        }

    @tf.function
    def train_step(self, x, *args):
        with tf.GradientTape() as tape:
            x_temp = tf.reshape(x, (-1, *self.input_shape[1:]))
            deform_out = self.deform_model(x_temp)
            b_s, h, w, c = deform_out.shape.as_list()
            deform_out = tf.reshape(
                deform_out, (self.batch_size, -1, h, w, c))
            convlstm_out = self.convlstm_model(deform_out)
            tr_loss = self.loss_fn(args[0], convlstm_out)
        trainable_vars = self.deform_model.trainable_variables + \
            self.convlstm_model.trainable_variables
        grads = tape.gradient(tr_loss, trainable_vars)
        self.optim.apply_gradients(zip(grads, trainable_vars))
        self.metrics["tr_loss"].update_state(tr_loss)
        self.metrics["tr_rec"].update_state(
            args[0], tf.nn.sigmoid(convlstm_out))
        self.metrics["tr_pre"].update_state(
            args[0], tf.nn.sigmoid(convlstm_out))

    @tf.function
    def val_step(self, x, *args):
        x_temp = tf.reshape(x, (-1, *self.input_shape[1:]))
        deform_out = self.deform_model(x_temp)
        b_s, h, w, c = deform_out.shape.as_list()
        deform_out = tf.reshape(deform_out, (1, -1, h, w, c))
        convlstm_out = self.convlstm_model(deform_out)
        val_loss = self.loss_fn(args[0], convlstm_out)
        self.metrics["vl_loss"].update_state(val_loss)
        self.metrics["vl_rec"].update_state(
            args[0], tf.nn.sigmoid(convlstm_out))
        self.metrics["vl_pre"].update_state(
            args[0], tf.nn.sigmoid(convlstm_out))
