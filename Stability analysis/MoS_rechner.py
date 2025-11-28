import numpy as np

def compute_mos(sto_lines):
    """
    Berechnet die MoS-Werte aus einer .sto-Datei und gibt die wichtigsten Kennwerte zurück.

    Parameter:
    - sto_lines (list of str): Inhalt der .sto-Datei (inkl. Header)

    Rückgabe:
    - mos_ap_min (float)
    - mos_ap_max (float)
    - mos_ap_mean (float)
    - bt_ap_min (float)
    - bt_ap_max (float)
    - bt_ap_mean (float)
    """

    for i, line in enumerate(sto_lines):
        if line.strip().lower() == 'endheader':
            header_end_idx = i
            break
    else:
        raise ValueError("End of header not found in the .sto file")

    column_names = sto_lines[header_end_idx + 1].split()

    required = ["com_x", "com_x_u", "leg1_r.cop_x", "leg1_r.grf_y", "com_y"]
    if not all(col in column_names for col in required):
        raise ValueError("Eine oder mehrere erforderliche Spalten fehlen in der .sto-Datei")

    idx = {col: column_names.index(col) for col in required}
    data = np.loadtxt(sto_lines[header_end_idx + 2:], dtype=float)

    com_x = data[:, idx["com_x"]][250:]
    com_x_u = data[:, idx["com_x_u"]][250:]
    com_y = data[:, idx["com_y"]][250:]
    cop_x = data[:, idx["leg1_r.cop_x"]][250:]
    grf_y = data[:, idx["leg1_r.grf_y"]][250:]

    # Gültige Werte (com_y > 0)
    valid_mask = com_y > 0
    com_x = com_x[valid_mask]
    com_x_u = com_x_u[valid_mask]
    com_y = com_y[valid_mask]
    cop_x = cop_x[valid_mask]
    grf_y = grf_y[valid_mask]

    g = 9.81
    omega = np.sqrt(g / com_y)
    xcom_x = com_x + com_x_u / omega

    mos_ap = cop_x - xcom_x
    with np.errstate(divide='ignore', invalid='ignore'):
        bt_ap = np.where(com_x_u != 0, mos_ap / com_x_u, np.inf)

    stance_mask = grf_y != 0
    mos_ap_stance = mos_ap[stance_mask]
    bt_ap_stance = bt_ap[stance_mask]

    return (
        float(np.min(mos_ap_stance)),
        float(np.max(mos_ap_stance)),
        float(np.mean(mos_ap_stance)),
        float(np.min(bt_ap_stance)),
        float(np.max(bt_ap_stance)),
        float(np.mean(bt_ap_stance))
    )
