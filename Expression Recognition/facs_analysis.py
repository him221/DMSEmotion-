# facs_analysis.py

facs_dict = {
    'AU1': '内眉毛提升',
    'AU2': '外眉毛提升',
    'AU4': '皱眉',
    'AU6': '眼轮匝肌收缩 (微笑)',
    'AU9': '鼻翼提肌 (厌恶)',
    'AU12': '嘴角上扬 (微笑)',
    'AU15': '嘴角下拉 (悲伤)',
    'AU20': '嘴唇拉伸',
    'AU25': '嘴唇分开',
    'AU26': '下巴下拉'
}

def get_facs_from_landmarks(landmarks):
    facs_results = []
    # 判断眉毛动作 (AU1, AU2, AU4)
    if landmarks[21][1] < landmarks[19][1]:
        facs_results.append('AU1')
    if landmarks[22][1] < landmarks[24][1]:
        facs_results.append('AU2')
    if abs(landmarks[21][0] - landmarks[22][0]) < 10:
        facs_results.append('AU4')

    # 判断眼部动作 (AU6)
    if landmarks[41][1] < landmarks[37][1]:
        facs_results.append('AU6')

    # 判断鼻部动作 (AU9)
    if landmarks[31][1] < landmarks[30][1]:
        facs_results.append('AU9')

    # 判断嘴部动作 (AU12, AU15, AU20, AU25, AU26)
    if landmarks[48][1] < landmarks[54][1]:
        facs_results.append('AU12')
    elif landmarks[48][1] > landmarks[54][1]:
        facs_results.append('AU15')

    if abs(landmarks[48][0] - landmarks[54][0]) > 15:
        facs_results.append('AU20')

    if landmarks[62][1] > landmarks[66][1]:
        facs_results.append('AU25')
    if landmarks[57][1] > landmarks[51][1]:
        facs_results.append('AU26')

    return facs_results
