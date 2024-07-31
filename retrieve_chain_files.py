def retrieve_chain_files(python_out_file,warmup=False,database_file=False):
	for l_i,line in enumerate(reversed(list(open(python_out_file)))):
		if warmup:
			if 'Saved warmup to' in line:
				return line.split('Saved warmup to')[1].strip()
		elif database_file:
			if 'Namespace(filein=' in line:
				return line.split("Namespace(filein='")[1].split("', p=")[0].strip()
		else:
			if 'Saving samples to' in line:
				return line.split('Saving samples to')[1].strip()
			if l_i>100: print('No Sample file Found');break