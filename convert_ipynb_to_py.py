import sys,json

def save_notebook_as_python_file(input_file, output_file):
    f = open(input_file, 'r') #input.ipynb
    j = json.load(f)
    of = open(output_file, 'w') #output.py
    if j["nbformat"] >=4:
            for i,cell in enumerate(j["cells"]):
                    of.write("#cell "+str(i)+"\n")
                    for line in cell["source"]:
                            of.write(line)
                    of.write('\n\n')
    else:
            for i,cell in enumerate(j["worksheets"][0]["cells"]):
                    of.write("#cell "+str(i)+"\n")
                    for line in cell["input"]:
                            of.write(line)
                    of.write('\n\n')

    of.close()