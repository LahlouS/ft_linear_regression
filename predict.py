import sys
from lib import LinearRegression, L2_average

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
    
    miles = model.zscore(miles, model.mean_expl, model.std_expl)
    pred = model.predict(miles)
    pred = model.scale(pred, model.mean_label, model.std_label)

    print('\nPREDICTION:', pred)

    pass
except Exception as e:
    if isinstance(e, ValueError):
        print(f'Error occurs converting {sys.argv[1]}')
        print(REGEX)
