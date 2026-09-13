import originpro as op
import numpy as np

acv_csv = r'D:\fscbm\experiments\20260824-150009_acv_s729\train_curves.csv'
scbm_csv = r'D:\fscbm\experiments\20260827-165033_scbm_s729\train_curves.csv'

def load(path):
    """读取 train_curves.csv 数据区，返回 (epoch, loss, acc)"""
    rows = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            p = ln.strip().split(',')
            if len(p) == 3 and p[0].isdigit():
                rows.append([float(x) for x in p])
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]

e1, l1, c1 = load(acv_csv)    # ACV-FSCBM
e2, l2, c2 = load(scbm_csv)   # SCBM

def place_legend(gl, lg, corner):
    """把图例放到指定角落（用轴坐标，不遮挡曲线）
    corner: 'tr'=右上, 'br'=右下, 'tl'=左上, 'bl'=左下"""
    xf = gl.get_float('x.from'); xt = gl.get_float('x.to')
    yf = gl.get_float('y.from'); yt = gl.get_float('y.to')
    dx = lg.get_float('dx'); dy = lg.get_float('dy')
    if corner == 'tr': lg.set_float('x', xt - dx/2); lg.set_float('y', yt - dy/2)
    elif corner == 'tl': lg.set_float('x', xf + dx/2); lg.set_float('y', yt - dy/2)
    elif corner == 'bl': lg.set_float('x', xf + dx/2); lg.set_float('y', yf + dy/2)
    else: lg.set_float('x', xt - dx/2); lg.set_float('y', yf + dy/2)  # br

def make_fig(xA, yA, xB, yB, ylab, lcorner):
    w = op.new_sheet('w')
    w.from_list(0, xA, 'x_acv'); w.from_list(1, yA, 'y_acv')
    w.from_list(2, xB, 'x_scbm'); w.from_list(3, yB, 'y_scbm')
    g = op.new_graph(template='line')
    gl = g[0]
    # ACV：实线，红色
    p1 = gl.add_plot(w, coly=1, colx=0)
    p1.color = 'red'
    p1.set_cmd('-w 1200')                    # 线宽 2.4pt（加粗，避免 Word 缩排后模糊）
    # SCBM：虚线，蓝色
    p2 = gl.add_plot(w, coly=3, colx=2)
    p2.color = 'blue'
    try:
        p2.set_cmd('-d 1', '-w 1200')
    except Exception:
        p2.set_cmd('-w 1200')
        print('若 SCBM 仍是实线，请双击曲线手动改为 Dash')
    gl.rescale()
    # 坐标轴标题：富文本 \b 加粗（字号在 Origin 中手动调整）
    gl.axis('x').title = '\\b(训练批次)'
    gl.axis('y').title = f'\\b({ylab})'
    # 刻度数字字号：x 下轴 / y 左轴 均为 24
    gl.lt_exec('xb.fsize = 24; yl.fsize = 24;')
    # 图例：文本 + 字号 + 定位到指定角落
    lg = gl.label('Legend')
    if lg is not None:
        lg.text = '\\l(1) ACV-FSCBM\n\\l(2) SCBM'
        lg.set_int('fsize', 22)
        place_legend(gl, lg, lcorner)
    return g

# 图1：损失变化 —— 曲线右上侧是空白，图例放右上
make_fig(e1, l1, e2, l2, '损失值', 'tr')
# 图2：准确率变化 —— ACV 终点在右上，图例放右下
make_fig(e1, c1, e2, c2, '准确率', 'br')

print('两张图已生成。选中图形窗口 -> 右键 Copy Page -> Word 粘贴')