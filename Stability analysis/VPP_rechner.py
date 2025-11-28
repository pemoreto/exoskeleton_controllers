import numpy as np
import matplotlib.pyplot as plt

def SS_detect(GRF_right, GRF_left):
    """
    Detect the Single Support Phase.
    Args:
        GRF_right(array): x,y GRF of the right leg
        GRF_left(array): x,y GRF of the right leg

    Returns:

    """
    # Calculate magnitude of the Ground Reaction Force (GRF)
    fMag_right = np.sqrt(GRF_right[:, 0]**2 + GRF_right[:, 1]**2)
    fMag_left = np.sqrt(GRF_left[:, 0]**2 + GRF_left[:, 1]**2)

    threshold = 0.005  # Detection threshold

    isLeftInContact = fMag_left >= threshold
    isRightInContact = fMag_right >= threshold

    # Single support phase: one leg in contact, the other not
    isSingleSupportLeft = isLeftInContact & ~isRightInContact
    isSingleSupportRight = isRightInContact & ~isLeftInContact

    # Detect transitions for left leg single support
    diffLeft = np.diff(np.concatenate([[0], isSingleSupportLeft.astype(int), [0]]))
    ST_left = np.where(diffLeft == 1)[0]
    EN_left = np.where(diffLeft == -1)[0] - 1

    # Detect transitions for right leg single support
    diffRight = np.diff(np.concatenate([[0], isSingleSupportRight.astype(int), [0]]))
    ST_right = np.where(diffRight == 1)[0]
    EN_right = np.where(diffRight == -1)[0] - 1

    return (ST_left, EN_left), (ST_right, EN_right)




def findVPP(GRF, CoP, CoM, plotFlag):
    fMag = np.sqrt(GRF[:, 0] ** 2 + GRF[:, 1] ** 2)
    ind = np.where(fMag > 0)[0]
    phi = np.pi / 2

    cCoP = CoP[ind, :2] - CoM[ind, :2]

    CoPx = cCoP[:, 0] * np.sin(phi) - cCoP[:, 1] * np.cos(phi)
    coPy = cCoP[:, 0] * np.cos(phi) + cCoP[:, 1] * np.sin(phi)
    nFx = GRF[ind, 0] * np.sin(phi) - GRF[ind, 1] * np.cos(phi)
    nFy = GRF[ind, 0] * np.cos(phi) + GRF[ind, 1] * np.sin(phi)

    vpp = plotvects(CoPx, coPy, nFx, nFy, 2, plotFlag)

    if plotFlag:
        plt.xlabel(r'$x~\scriptstyle(m)$', fontsize=18)
        plt.ylabel(r'$y~\scriptstyle(m)$', fontsize=18)

    return vpp


def calculate_VPP(fname):
    # --- Read .sto file ---
    with open(fname, 'r') as f:
        sto_lines = f.readlines()

    for i, line in enumerate(sto_lines):
        if line.strip().lower() == 'endheader':
            header_end_idx = i
            break
    else:
        raise ValueError("End of header not found in the .sto file")

    column_names = sto_lines[header_end_idx + 1].split()

    required = [
        "leg1_r.grf_x", "leg1_r.grf_y",
        "leg0_l.grf_x", "leg0_l.grf_y",
        "leg1_r.cop_x", "leg1_r.cop_y",
        "leg0_l.cop_x", "leg0_l.cop_y",
        "com_x", "com_y", "com_z",
        "Effort.penalty"
    ]

    if not all(col in column_names for col in required):
        missing = [col for col in required if col not in column_names]
        raise ValueError(f"Missing required columns in the .sto file: {missing}")

    idx = {col: column_names.index(col) for col in required}
    data = np.loadtxt(sto_lines[header_end_idx + 2:], dtype=float)

    # Extract data
    GRF_r = data[:, [idx["leg1_r.grf_x"], idx["leg1_r.grf_y"]]]
    GRF_l = data[:, [idx["leg0_l.grf_x"], idx["leg0_l.grf_y"]]]
    CoP_r = data[:, [idx["leg1_r.cop_x"], idx["leg1_r.cop_y"]]]
    CoP_l = data[:, [idx["leg0_l.cop_x"], idx["leg0_l.cop_y"]]]
    CoM = data[:, [idx["com_x"], idx["com_y"], idx["com_z"]]]
    EffortPenalty = data[:, idx["Effort.penalty"]]

    # --- Detect single support phases ---
    (ST_left, EN_left), (ST_right, EN_right) = SS_detect(GRF_r, GRF_l)

    results = {
        'right': {'ST': ST_right, 'EN': EN_right, 'cycles': [], 'R2': [], 'EffortPenalty': []},
        'left': {'ST': ST_left, 'EN': EN_left, 'cycles': [], 'R2': [], 'EffortPenalty': []}
    }

    for leg in ['right', 'left']:
        ST = results[leg]['ST']
        EN = results[leg]['EN']

        numPhases = len(ST)
        numToAnalyze = min(5, numPhases)
        print(f'{leg.upper()} LEG - numToAnalyze: {numToAnalyze} / numPhases: {numPhases}')

        last_ST = ST[-numToAnalyze:]
        last_EN = EN[-numToAnalyze:]

        for i in range(numToAnalyze):
            idx_range = range(last_ST[i], last_EN[i] + 1)

            cycle = {
                'GRF_r': GRF_r[idx_range, :],
                'GRF_l': GRF_l[idx_range, :],
                'CoM': CoM[idx_range, :],
                'CoP_r': CoP_r[idx_range, :],
                'CoP_l': CoP_l[idx_range, :],
                'EffortPenalty': EffortPenalty[idx_range]
            }

            if leg == 'right':
                GRF = cycle['GRF_r']
                CoP = cycle['CoP_r']
            else:
                GRF = cycle['GRF_l']
                CoP = cycle['CoP_l']

            VPP = findVPP(GRF, CoP, cycle['CoM'], plotFlag=False)
            cycle[f'VPP_{leg}_leg'] = VPP
            results[leg]['EffortPenalty'].append(np.mean(cycle['EffortPenalty']))

            # Compute R²
            FX = GRF[:, 0]
            FY = GRF[:, 1]
            CoPx = CoP[:, 0]
            CoPy = CoP[:, 1]
            CoMx = cycle['CoM'][:, 0]
            CoMy = cycle['CoM'][:, 1]

            new_CoPx = CoPx - CoMx
            new_CoPy = CoPy - CoMy

            R2 = R2cal(FX, FY, new_CoPx, new_CoPy, VPP)
            results[leg]['R2'].append(R2)

            results[leg]['cycles'].append(cycle)

    return results


def R2cal(FX, FY, posX, posY, vpp):
    # Correct arctan2 usage: arctan2(dy, dx)
    ang_exp = np.arctan2(FY, FX)
    ang_est = np.arctan2(vpp[1] - posY, vpp[0] - posX)

    # Wrap angles to [-π, π] difference
    ang_diff = np.angle(np.exp(1j * (ang_exp - ang_est)))

    m_exp = np.mean(ang_exp)
    ang_exp_diff = np.angle(np.exp(1j * (ang_exp - m_exp)))

    R2 = (1 - np.sum(ang_diff**2) / np.sum(ang_exp_diff**2)) * 100
    return R2



def plotvects(posX, posY, FX, FY, l, plotFlag):
    x = [0, 0]
    y = [0, 0]
    step = 1
    L = np.max(np.sqrt(FX**2 + FY**2))
    count = 0
    Y2 = np.zeros_like(posY)
    nFX = np.zeros_like(FX)
    nFY = np.zeros_like(FY)

    nonzero_FY_count = len(np.where(FY > 0)[0])
    if nonzero_FY_count == 0:
        nonzero_FY_count = 1  # prevent division by zero

    for i in range(len(posX)):
        x[0] = posX[i]
        y[0] = posY[i]

        nFX[i] = FX[i] * l / L
        x[1] = x[0] + nFX[i]
        nFY[i] = FY[i] * l / L
        y[1] = y[0] + nFY[i]
        Y2[i] = y[1]

        if FY[i] > 0:
            count += 1

        if (i % step == 0 or step == 1) and plotFlag and (i % 2 == 1):
            grey = count * 0.95 / nonzero_FY_count
            plt.plot(x, y, linewidth=2, color=[grey, grey, grey])
            plt.gca().set_aspect('equal', adjustable='box')

    tau0 = np.dot(posY, nFX) - np.dot(posX, nFY)
    Px = np.sum(nFX)
    Py = np.sum(nFY)

    a, b, c = 0, 0, 0
    if Px != 0 or Py != 0:
        dist = tau0 / np.sqrt(Px**2 + Py**2)
        if Px == 0:
            a, b, c = 0, 1, dist
        elif Py == 0:
            a, b, c = -1, 0, dist
        else:
            a = -Py / Px
            b = 1
            c = np.sign(Px) * dist * np.sqrt(1 + a**2)

        ymin = np.min(posY)
        ymax = np.max(Y2)

        if plotFlag:
            plotLine(a, b, c, ymin, ymax)

    alpha = nFX * (posY - c) - nFY * posX
    betta = nFY + a * nFX
    rx = -np.linalg.lstsq(betta[:, np.newaxis], alpha, rcond=None)[0][0]
    ry = -a * rx + c
    J = nFX * (posY - ry) - nFY * (posX - rx)
    J = np.dot(J, J)
    vpp = np.array([rx, ry])

    if plotFlag:
        plt.plot(0, 0, 'og', linewidth=5)
        plt.plot(rx, ry, 'or', linewidth=5)

    return vpp


def plotLine(a, b, c, ymin, ymax):
    if a != 0 or b != 0:
        if a == 0:
            if b != 1:
                c = c / b
            Y = [c, c]
            X = [ymin, ymax]
        elif b == 0:
            if a != -1:
                c = c / a
            Y = [ymin, ymax]
            X = [-c, -c]
        else:
            Y = [ymin, ymax]
            X = [(-b * y + c) / a for y in Y]

        plt.plot(X, Y, 'k', linewidth=2)


