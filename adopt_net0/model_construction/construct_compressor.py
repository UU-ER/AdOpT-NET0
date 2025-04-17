from adopt_net0.components.compressors.compressor import Compressor


def construct_compressor_block(b_compr, data: dict, set_t_full, set_t_clustered):
    """
    Construct compressor block and performs

    :param b_tec: pyomo block with compressor model
    :param dict data: data containing model configuration
    :param set_t_full: pyomo set containing timesteps
    :param set_t_clustered: pyomo set containing clustered timesteps
    :return: pyomo block with compressor model
    """
    compr = b_compr.index()
    carrier, tec1, tec2 = compr

    compressor = data["compression"][carrier].get((tec1, tec2))

    if compressor:
        p_in, p_out = compressor  # Ottieni p_in e p_out
        compressor = {
            "carrier": carrier,
            "tec1": tec1,
            "tec2": tec2,
            "p_in": p_in,
            "p_out": p_out,
        }

    b_compr = Compressor(compressor).construct_compressor_model(
        b_compr, data, set_t_full, set_t_clustered
    )

    return b_compr
