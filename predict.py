import sys

print(sys.argv)
REGEX = 'usage: python predict.py [ miles: integer ]'
if len(sys.argv) != 2:
    print('Error occurs: predictor wait for only one parameter')
    print(REGEX)

try:
    
    pass
except Exception as e:
    if isinstance(e, ValueError):
        print('Error occurs converting argv[1]')
        print(REGEX)
