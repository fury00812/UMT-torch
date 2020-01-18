#from .transformer import TransformerEncoder, TransformerDecoder 
#from .discriminator import Discriminator
#from .lm import LM


def build_transformer_enc_dec(params):
    """
    Build transformer encoder and decoder
    """

    # set parameters of transformer encoder
    params.encoder_embed_dim = params.emb_dim
    params.encoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.encoder_layers = params.n_enc_layers

    # set parameters of transformer decoder
    params.decoder_embed_dim = params.emb_dim
    params.decoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.decoder_layers = params.n_dec_layers

    # build transformer encoder
    logger.info("============ Building transformer attention model - Encoder ...")
    encoder = TransformerEncoder(params)
    logger.info("")

    # build transformer decoder
    logger.info("============ Building transformer attention model - Decoder ...")
    decoder = TransformerDecoder(params, encoder)
    logger.info("")

    return encoder, decoder


def build_model(params, data, cuda=True):
    """
    Build machine translation model - encoder, decoder, and language model
    """
    # encoder / decoder
    encoder, decoder = build_transformer_enc_dec(params)

    # loss function for decoder reconstruction
    

    # language model
