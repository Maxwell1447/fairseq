import numpy as np

PATH_TO_SAVE="/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-domain/infer-logits/"


if __name__ == "__main__":
    logits_y = np.load(PATH_TO_SAVE+"ECB.del.logits.unsquashed.0.npy")
    logits_x = np.load(PATH_TO_SAVE+"ECB.del.logits.0.npy")
    y = np.load(PATH_TO_SAVE+"ECB.del.toks.unsquashed.0.npy")
    x = np.load(PATH_TO_SAVE+"ECB.del.toks.0.npy")



    assert (x == y).all()

    mask_x = (x != 1)
    mask_y = mask_x

    xxx = logits_x[mask_x]
    yyy = logits_y[mask_y]

    print("normalization squashed =", (np.exp(xxx).sum(-1) == 1).sum() / len(xxx.sum(-1)))
    print("normalization unsquashed =", (np.exp(yyy).sum(-1) == 1).sum() / len(yyy.sum(-1)))

    left_x = xxx[...]
    left_y = yyy[...]
    # left_x = np.exp(xxx)[..., 0]
    # left_y = np.exp(yyy)[..., 0]

    print("mean diff prob =", np.abs(left_x - left_y).mean())

    # for ext in ["-tok", "-proj", "-pos", "-seq", ""]:
    # for ext in [""]:
    #     print(ext)
    #     embed_y = np.load(PATH_TO_SAVE+f"ECB.del.embed{ext}.0.npy")
    #     embed_x = np.load(PATH_TO_SAVE+f"ECB.del.embed{ext}.unsquashed.0.npy")

    #     exx = embed_x[mask_x]
    #     eyy = embed_y[mask_y]

    #     print("mean diff embed =", np.abs(exx[..., 0] - eyy[..., 0]).mean())


    # seq_y = np.load(PATH_TO_SAVE+"ECB.del.seq.0.npy")
    # seq_x = np.load(PATH_TO_SAVE+"ECB.del.seq.unsquashed.0.npy")

    
    # print("unsquashed")
    # print(seq_x.shape)
    # print(seq_x[..., 0])
    # print("squashed")
    # print(seq_y.shape)
    # print(seq_y[..., 0])

    # exx = seq_x[mask_x]
    # eyy = seq_y[mask_y]


    # print("mean diff seq =", np.abs(exx[..., 0] - eyy[..., 0]).mean())