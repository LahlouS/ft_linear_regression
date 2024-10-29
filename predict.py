import sys
from lib import LinearRegression, L2_average, create_viz_folder
from visualizer import Visualizer
import numpy as np

REGEX = 'usage: python predict.py [ miles: integer ]'
if len(sys.argv) != 1:
    print('Error occurs: predict take only one parameter')
    print(REGEX)
    sys.exit()

try:
    miles = input('enter a mile >>')
    miles = int(miles)

    model = LinearRegression()

    try:
        model.read_weights('model.ckpt')
    except Exception as e:
        print('Error occurs locating model weights file\nWarning: running prediction with weight set to 0')
        model.a = 0
        model.b = 0

    norm_miles = model.zscore(miles, model.mean_expl, model.std_expl)
    pred = model.predict(norm_miles)
    pred = model.scale(pred, model.mean_label, model.std_label)


    viz_folder = 'visualization/'
    create_viz_folder(viz_folder)
    x = model.zscore(22899, model.mean_expl, model.std_expl)
    x2 = model.zscore(139800, model.mean_expl, model.std_expl)

    # just to draw the line
    ref_point1 = [22899, model.scale(model.predict(x), model.mean_label, model.std_label)]
    ref_point2 = [139800, model.scale(model.predict(x2), model.mean_label, model.std_label)]

    point = [miles, pred]
    viz = Visualizer(model.datas)
    viz.raw_data(np.array([point])).line(ref_point1, ref_point2).to_file(viz_folder + 'proxy.html')
    print('\nPREDICTION:', pred)

except Exception as e:
    if isinstance(e, ValueError):
        print(f'Error occurs converting {sys.argv[1]}')
        print(REGEX)
    else:
        raise e
