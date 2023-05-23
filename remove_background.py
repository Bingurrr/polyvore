from rembg import remove

input_path = '/home/park/capstone/polyvore/results/0_150153178_1.jpg'
output_path = '/home/park/capstone/polyvore/results/remove_0_150153178_1.jpg'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
